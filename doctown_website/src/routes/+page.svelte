<script lang="ts">
	import { onMount } from 'svelte';

	// Types
	interface Citation {
		id: number;
		file_path: string;
		chunk_index: number;
		quote: string;
		start_char: number | null;
		end_char: number | null;
	}

	interface AnswerResponse {
		answer: string;
		citations: Citation[];
		confidence: string;
		sources_retrieved: number;
		sources_used: number;
	}

	interface Progress {
		phase: string;
		current: number;
		total: number;
	}

	interface Stats {
		total_files: number;
		total_chunks: number;
		total_vectors: number;
		total_size_bytes: number;
	}

	interface Timing {
		phase_elapsed: number | null;
		pipeline_elapsed: number | null;
		phase_times: Record<string, number>;
	}

	interface ServerState {
		stage: string;
		progress: Progress;
		stats: Stats | null;
		error: string | null;
		timing: Timing | null;
	}

	interface StagedFile {
		name: string;
		size: number;
	}

	interface StagedFilesResponse {
		files: StagedFile[];
		total_size: number;
		file_count: number;
	}

	// State
	let stage = $state<string>('idle');
	let directoryPath = $state('');
	let debugLog = $state<string[]>([]);

	function log(msg: string) {
		console.log(msg);
		debugLog = [...debugLog.slice(-10), msg];
	}

	// File staging state
	let stagedFiles = $state<StagedFile[]>([]);
	let stagedTotalSize = $state(0);
	let isUploading = $state(false);
	let isDragOver = $state(false);
	let showAdvanced = $state(false);
	let searchQuery = $state('');
	let answerResponse = $state<AnswerResponse | null>(null);
	let progress = $state<Progress>({ phase: '', current: 0, total: 0 });
	let stats = $state<Stats | null>(null);
	let error = $state<string | null>(null);
	let isAsking = $state(false);
	let timing = $state<Timing | null>(null);
	let finalPipelineTime = $state<number | null>(null);

	// Query timing
	let queryStartTime = $state<number | null>(null);
	let queryElapsed = $state<number | null>(null);
	let queryTimerInterval = $state<ReturnType<typeof setInterval> | null>(null);

	// Processing timing (real-time local counter)
	let processingStartTime = $state<number | null>(null);
	let phaseStartTime = $state<number | null>(null);
	let localPhaseElapsed = $state<number>(0);
	let localPipelineElapsed = $state<number>(0);
	let processingTimerInterval = $state<ReturnType<typeof setInterval> | null>(null);

	// Derived
	let progressPercent = $derived(
		progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0
	);

	let isProcessing = $derived(
		stage === 'freezing' || stage === 'chunking' || stage === 'embedding' || stage === 'summarizing'
	);

	let canSearch = $derived(stage === 'ready' && !isAsking);

	// SSE connection
	onMount(() => {
		const eventSource = new EventSource('/api/events');

		eventSource.onmessage = (event) => {
			const data: ServerState = JSON.parse(event.data);
			const prevStage = stage;
			stage = data.stage;
			progress = data.progress;
			if (data.stats) {
				stats = data.stats;
			}
			error = data.error;
			if (data.timing) {
				timing = data.timing;
				// Capture final pipeline time when transitioning to ready
				if (prevStage !== 'ready' && data.stage === 'ready' && data.timing.pipeline_elapsed) {
					finalPipelineTime = data.timing.pipeline_elapsed;
				}
				// Also calculate from phase_times if we're in ready state and don't have finalPipelineTime yet
				if (data.stage === 'ready' && !finalPipelineTime && data.timing.phase_times) {
					const totalFromPhases = Object.values(data.timing.phase_times).reduce((sum, t) => sum + t, 0);
					if (totalFromPhases > 0) {
						finalPipelineTime = totalFromPhases;
					}
				}
			}

			// Handle processing timer
			const isNowProcessing = ['freezing', 'chunking', 'embedding', 'summarizing'].includes(data.stage);
			const wasProcessing = ['freezing', 'chunking', 'embedding', 'summarizing'].includes(prevStage);
			const timerRunning = processingTimerInterval !== null;

			// Start timer when processing begins OR if we connect while already processing
			if (isNowProcessing && (!wasProcessing || !timerRunning)) {
				// Use server's elapsed time if available to sync up
				const serverPipelineElapsed = data.timing?.pipeline_elapsed ?? 0;
				const serverPhaseElapsed = data.timing?.phase_elapsed ?? 0;

				processingStartTime = performance.now() - (serverPipelineElapsed * 1000);
				phaseStartTime = performance.now() - (serverPhaseElapsed * 1000);
				localPipelineElapsed = serverPipelineElapsed;
				localPhaseElapsed = serverPhaseElapsed;

				if (processingTimerInterval) clearInterval(processingTimerInterval);
				processingTimerInterval = setInterval(() => {
					if (processingStartTime !== null) {
						localPipelineElapsed = (performance.now() - processingStartTime) / 1000;
					}
					if (phaseStartTime !== null) {
						localPhaseElapsed = (performance.now() - phaseStartTime) / 1000;
					}
				}, 100);
			}

			// Reset phase timer on phase change (but only if timer is running)
			if (isNowProcessing && wasProcessing && data.stage !== prevStage && timerRunning) {
				phaseStartTime = performance.now();
				localPhaseElapsed = 0;
			}

			// Stop timer when processing ends
			if (!isNowProcessing && wasProcessing) {
				if (processingTimerInterval) {
					clearInterval(processingTimerInterval);
					processingTimerInterval = null;
				}
			}
		};

		eventSource.onerror = () => {
			console.error('SSE connection error');
		};

		return () => {
			eventSource.close();
			// Clean up timers
			if (queryTimerInterval) {
				clearInterval(queryTimerInterval);
			}
			if (processingTimerInterval) {
				clearInterval(processingTimerInterval);
			}
		};
	});

	// Handlers
	async function handleProcess() {
		error = null;
		answerResponse = null;
		finalPipelineTime = null;
		timing = null;

		try {
			let body: { path?: string; use_staged?: boolean };

			if (stagedFiles.length > 0) {
				// Process staged files
				body = { use_staged: true };
			} else if (directoryPath.trim()) {
				// Process filesystem path (advanced mode)
				body = { path: directoryPath.trim() };
			} else {
				return;
			}

			const response = await fetch('/api/process', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			});

			if (!response.ok) {
				const data = await response.json();
				error = data.detail || 'Failed to start processing';
			}
			// SSE will update the stage automatically
		} catch (e) {
			error = e instanceof Error ? e.message : 'Network error';
		}
	}

	// File upload handlers
	async function uploadFiles(files: FileList | File[]) {
		if (!files || files.length === 0) return;

		isUploading = true;
		error = null;

		try {
			log('uploadFiles started');

			const formData = new FormData();
			let totalSize = 0;
			for (let i = 0; i < files.length; i++) {
				const file = files[i];
				log(`append ${i}: ${file.name}`);
				formData.append('files', file);
				totalSize += file.size;
			}
			log(`sending ${files.length} files (${totalSize} bytes)`);

			// Use absolute URL to ensure we hit the right server
			const baseUrl = window.location.origin;
			log(`POST to ${baseUrl}/api/upload`);

			const response = await fetch(`${baseUrl}/api/upload`, {
				method: 'POST',
				body: formData
			});
			log(`response: ${response.status} ${response.statusText}`);

			if (!response.ok) {
				const data = await response.json();
				error = data.detail || 'Upload failed';
				return;
			}

			const data: StagedFilesResponse = await response.json();
			stagedFiles = data.files;
			stagedTotalSize = data.total_size;
			console.log(`[upload] Success: ${data.file_count} files staged`);
		} catch (e) {
			log(`error: ${e}`);
			error = e instanceof Error ? e.message : 'Upload failed';
		} finally {
			isUploading = false;
		}
	}

	async function removeFile(fileName: string) {
		try {
			const response = await fetch(`/api/staged/${encodeURIComponent(fileName)}`, {
				method: 'DELETE'
			});

			if (response.ok) {
				stagedFiles = stagedFiles.filter(f => f.name !== fileName);
				stagedTotalSize = stagedFiles.reduce((sum, f) => sum + f.size, 0);
			}
		} catch (e) {
			console.error('Failed to remove file:', e);
		}
	}

	async function clearAllFiles() {
		try {
			const response = await fetch('/api/staged', {
				method: 'DELETE'
			});

			if (response.ok) {
				stagedFiles = [];
				stagedTotalSize = 0;
			}
		} catch (e) {
			console.error('Failed to clear files:', e);
		}
	}

	function handleFileInput(e: Event) {
		try {
			log('handleFileInput triggered');
			const input = e.target as HTMLInputElement;
			log(`files: ${input.files?.length ?? 'null'}`);
			if (input.files && input.files.length > 0) {
				log(`uploading ${input.files.length} files`);
				uploadFiles(input.files);
			}
		} catch (err) {
			log(`handleFileInput error: ${err}`);
			error = err instanceof Error ? err.message : 'Failed to read files';
		}
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		isDragOver = false;

		if (e.dataTransfer?.files) {
			uploadFiles(e.dataTransfer.files);
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		isDragOver = true;
	}

	function handleDragLeave(e: DragEvent) {
		e.preventDefault();
		isDragOver = false;
	}

	async function handleAsk() {
		if (!searchQuery.trim() || !canSearch) return;

		isAsking = true;
		error = null;

		// Start query timer
		queryStartTime = performance.now();
		queryElapsed = 0;
		if (queryTimerInterval) {
			clearInterval(queryTimerInterval);
		}
		queryTimerInterval = setInterval(() => {
			if (queryStartTime !== null) {
				queryElapsed = (performance.now() - queryStartTime) / 1000;
			}
		}, 50); // Update every 50ms for smooth display

		try {
			const response = await fetch('/api/ask', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ query: searchQuery.trim() })
			});

			if (!response.ok) {
				const data = await response.json();
				error = data.detail || 'Failed to get answer';
				return;
			}

			answerResponse = await response.json();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Network error';
		} finally {
			isAsking = false;
			// Stop timer and freeze at final value
			if (queryTimerInterval) {
				clearInterval(queryTimerInterval);
				queryTimerInterval = null;
			}
			if (queryStartTime !== null) {
				queryElapsed = (performance.now() - queryStartTime) / 1000;
			}
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			if (stage === 'idle' || stage === 'error') {
				if ((stagedFiles.length > 0 || directoryPath.trim()) && !isProcessing && !isUploading) {
					handleProcess();
				}
			} else if (stage === 'ready') {
				if (searchQuery.trim() && canSearch) {
					handleAsk();
				}
			}
		}
	}

	function getFileIcon(fileName: string): string {
		const ext = fileName.split('.').pop()?.toLowerCase() || '';
		const codeExts = ['js', 'ts', 'jsx', 'tsx', 'py', 'rb', 'go', 'rs', 'java', 'c', 'cpp', 'h', 'cs'];
		const docExts = ['md', 'txt', 'rst', 'doc', 'docx', 'pdf'];
		const dataExts = ['json', 'yaml', 'yml', 'xml', 'csv', 'toml'];
		const webExts = ['html', 'css', 'scss', 'sass', 'vue', 'svelte'];

		if (codeExts.includes(ext)) return 'code';
		if (docExts.includes(ext)) return 'doc';
		if (dataExts.includes(ext)) return 'data';
		if (webExts.includes(ext)) return 'web';
		return 'file';
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function formatTime(seconds: number): string {
		if (seconds < 60) {
			return `${seconds.toFixed(1)}s`;
		}
		const mins = Math.floor(seconds / 60);
		const secs = seconds % 60;
		return `${mins}m ${secs.toFixed(0)}s`;
	}

	function getPhaseLabel(phase: string): string {
		switch (phase) {
			case 'freezing': return 'Reading files';
			case 'chunking': return 'Chunking text';
			case 'embedding': return 'Generating embeddings';
			case 'summarizing': return 'Generating summaries';
			default: return phase;
		}
	}

	function getConfidenceColor(confidence: string): string {
		switch (confidence) {
			case 'high': return 'text-green-400';
			case 'medium': return 'text-yellow-400';
			case 'low': return 'text-orange-400';
			default: return 'text-red-400';
		}
	}

	function getConfidenceIcon(confidence: string): string {
		switch (confidence) {
			case 'high': return '++';
			case 'medium': return '+';
			case 'low': return '~';
			default: return '-';
		}
	}
</script>

<main class="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex flex-col items-center px-4 py-12">
	<div class="w-full max-w-3xl flex flex-col items-center gap-6">
		<!-- Logo/Title -->
		<h1 class="text-4xl font-bold text-white tracking-tight">
			Doctown
		</h1>
		<p class="text-white/60 text-base -mt-3">
			Instant answers from your documents
		</p>

		<!-- Error Display -->
		{#if error}
			<div class="w-full bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-200 text-sm">
				{error}
			</div>
		{/if}

		<!-- File Upload UI (show when idle) -->
		{#if stage === 'idle' || stage === 'error'}
			<div class="w-full mt-4 space-y-4">
				<!-- Drop zone -->
				<div
					class="relative border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer
						{isDragOver ? 'border-blue-400 bg-blue-500/10' : 'border-white/20 hover:border-white/40 hover:bg-white/5'}"
					ondrop={handleDrop}
					ondragover={handleDragOver}
					ondragleave={handleDragLeave}
					role="button"
					tabindex="0"
					onkeydown={(e) => e.key === 'Enter' && document.getElementById('file-input')?.click()}
					onclick={() => document.getElementById('file-input')?.click()}
				>
					{#if isUploading}
						<div class="flex flex-col items-center gap-3">
							<svg class="h-8 w-8 animate-spin text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
							<span class="text-white/60">Uploading files...</span>
						</div>
					{:else}
						<div class="flex flex-col items-center gap-3">
							<svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
							</svg>
							<div>
								<p class="text-white/70 text-lg">Drop files here or tap to browse</p>
								<p class="text-white/40 text-sm mt-1">Upload documents, code, or any text files</p>
							</div>
						</div>
					{/if}

					<!-- Hidden file inputs -->
					<input
						id="file-input"
						type="file"
						multiple
						class="hidden"
						onchange={handleFileInput}
					/>
				</div>

				<!-- Folder upload button (desktop browsers) -->
				<div class="flex gap-3 justify-center">
					<label class="px-4 py-2 bg-white/10 hover:bg-white/15 rounded-lg text-white/70 text-sm cursor-pointer transition-colors">
						<input
							type="file"
							multiple
							webkitdirectory
							class="hidden"
							onchange={handleFileInput}
						/>
						Upload Folder
					</label>
				</div>

				<!-- Staged files list -->
				{#if stagedFiles.length > 0}
					<div class="bg-white/5 rounded-xl p-4 space-y-2">
						<div class="flex items-center justify-between mb-3">
							<span class="text-white/60 text-sm">
								{stagedFiles.length} file{stagedFiles.length !== 1 ? 's' : ''} staged ({formatSize(stagedTotalSize)})
							</span>
							<button
								onclick={clearAllFiles}
								class="text-red-400/70 hover:text-red-400 text-xs transition-colors"
							>
								Clear all
							</button>
						</div>

						<div class="max-h-48 overflow-y-auto space-y-1">
							{#each stagedFiles as file}
								<div class="flex items-center gap-3 py-1.5 px-2 rounded-lg hover:bg-white/5 group">
									<!-- File icon -->
									<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-white/40 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										{#if getFileIcon(file.name) === 'code'}
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
										{:else if getFileIcon(file.name) === 'doc'}
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
										{:else if getFileIcon(file.name) === 'data'}
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
										{:else}
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
										{/if}
									</svg>

									<!-- File name -->
									<span class="flex-1 text-white/70 text-sm truncate" title={file.name}>
										{file.name}
									</span>

									<!-- File size -->
									<span class="text-white/30 text-xs shrink-0">
										{formatSize(file.size)}
									</span>

									<!-- Remove button -->
									<button
										onclick={() => removeFile(file.name)}
										class="opacity-0 group-hover:opacity-100 text-white/30 hover:text-red-400 transition-all p-1"
										title="Remove file"
									>
										<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
										</svg>
									</button>
								</div>
							{/each}
						</div>

						<!-- Process button -->
						<button
							onclick={handleProcess}
							disabled={isProcessing || isUploading}
							class="w-full mt-3 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors"
						>
							Process {stagedFiles.length} file{stagedFiles.length !== 1 ? 's' : ''}
						</button>
					</div>
				{/if}

				<!-- Advanced: Path input (collapsed by default) -->
				<div class="pt-2">
					<button
						onclick={() => showAdvanced = !showAdvanced}
						class="text-white/30 hover:text-white/50 text-xs flex items-center gap-1 transition-colors mx-auto"
					>
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 transition-transform {showAdvanced ? 'rotate-90' : ''}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
						</svg>
						Advanced: Load from server path
					</button>

					{#if showAdvanced}
						<div class="mt-3 flex gap-3">
							<input
								type="text"
								bind:value={directoryPath}
								onkeydown={handleKeydown}
								placeholder="/path/to/your/project"
								disabled={isProcessing || stagedFiles.length > 0}
								class="flex-1 px-4 py-3 text-sm bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all disabled:opacity-50"
							/>
							<button
								onclick={handleProcess}
								disabled={!directoryPath.trim() || isProcessing || stagedFiles.length > 0}
								class="px-5 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 disabled:cursor-not-allowed text-white text-sm font-medium rounded-xl transition-colors"
							>
								Load
							</button>
						</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Progress Bar (show during processing) -->
		{#if isProcessing}
			<div class="w-full bg-white/5 rounded-xl p-6 mt-4">
				<div class="flex items-center justify-between mb-3">
					<span class="text-white/80 font-medium">
						{getPhaseLabel(stage)}...
					</span>
					<div class="flex items-center gap-3">
						<span class="text-blue-400 text-sm font-mono tabular-nums">
							{formatTime(localPhaseElapsed)}
						</span>
						<span class="text-white/60 text-sm">
							{progress.current} / {progress.total}
						</span>
					</div>
				</div>
				<div class="w-full bg-white/10 rounded-full h-2 overflow-hidden">
					<div
						class="h-full bg-blue-500 transition-all duration-300 ease-out"
						style="width: {progressPercent}%"
					></div>
				</div>

				<!-- Completed phases timing + total elapsed -->
				<div class="flex flex-wrap items-center gap-x-4 gap-y-1 mt-3 text-xs">
					{#if timing?.phase_times && Object.keys(timing.phase_times).length > 0}
						{#each Object.entries(timing.phase_times) as [phase, duration]}
							<span class="text-white/40">
								{getPhaseLabel(phase)}: <span class="text-green-400 font-mono">{formatTime(duration)}</span>
							</span>
						{/each}
					{/if}
					<span class="text-white/30 ml-auto">
						total: <span class="text-blue-400/70 font-mono tabular-nums">{formatTime(localPipelineElapsed)}</span>
					</span>
				</div>

				<p class="text-white/40 text-sm mt-2">
					{#if stage === 'summarizing'}
						Generating AI summaries for better answers...
					{:else if stage === 'embedding'}
						Creating semantic vectors...
					{:else}
						Processing your files...
					{/if}
				</p>
			</div>
		{/if}

		<!-- Search Bar (show when ready) - Raycast-style -->
		{#if stage === 'ready'}
			<div class="w-full mt-4">
				<div class="relative">
					<div class="absolute left-5 top-1/2 -translate-y-1/2 text-white/40">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
						</svg>
					</div>
					<input
						type="text"
						bind:value={searchQuery}
						onkeydown={handleKeydown}
						placeholder="Ask anything about your documents..."
						disabled={isAsking}
						autofocus
						class="w-full pl-14 pr-6 py-5 text-lg bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
					/>
					{#if isAsking}
						<div class="absolute right-5 top-1/2 -translate-y-1/2 flex items-center gap-2">
							<span class="text-blue-400 text-sm font-mono tabular-nums">
								{queryElapsed !== null ? formatTime(queryElapsed) : ''}
							</span>
							<svg class="h-5 w-5 animate-spin text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						</div>
					{/if}
				</div>
				<p class="text-white/30 text-xs mt-2 text-center">Press Enter to search</p>
			</div>
		{/if}

		<!-- Answer Display -->
		{#if answerResponse}
			<div class="w-full mt-2 space-y-4">
				<!-- Main Answer Card -->
				<div class="bg-white/[0.08] border border-white/10 rounded-2xl p-6">
					<!-- Answer text -->
					<p class="text-white text-lg leading-relaxed">
						{answerResponse.answer}
					</p>

					<!-- Confidence indicator -->
					<div class="mt-4 flex items-center gap-2 text-sm">
						<span class="{getConfidenceColor(answerResponse.confidence)} font-mono">
							{getConfidenceIcon(answerResponse.confidence)}
						</span>
						<span class="text-white/50">
							{answerResponse.confidence} confidence
						</span>
						<span class="text-white/30">|</span>
						<span class="text-white/40">
							{answerResponse.sources_used} of {answerResponse.sources_retrieved} sources used
						</span>
						{#if queryElapsed !== null}
							<span class="text-white/30">|</span>
							<span class="text-white/40">
								<span class="text-blue-400 font-mono">{formatTime(queryElapsed)}</span>
							</span>
						{/if}
					</div>
				</div>

				<!-- Citations -->
				{#if answerResponse.citations.length > 0}
					<div class="space-y-2">
						<h3 class="text-white/50 text-xs font-medium uppercase tracking-wide px-1">Sources</h3>
						{#each answerResponse.citations as citation}
							<div class="bg-white/[0.04] border border-white/[0.08] rounded-xl p-4 hover:bg-white/[0.06] transition-colors group">
								<div class="flex items-start gap-3">
									<!-- Citation number -->
									<span class="text-blue-400 font-mono text-sm font-medium shrink-0">
										[{citation.id}]
									</span>
									<div class="flex-1 min-w-0">
										<!-- File path -->
										<div class="flex items-center gap-2 mb-2">
											<span class="text-blue-300 text-sm font-medium truncate">
												{citation.file_path}
											</span>
											{#if citation.start_char !== null}
												<span class="text-white/30 text-xs">
													chars {citation.start_char}-{citation.end_char}
												</span>
											{:else}
												<span class="text-white/30 text-xs">
													chunk {citation.chunk_index}
												</span>
											{/if}
										</div>
										<!-- Quote -->
										<p class="text-white/60 text-sm leading-relaxed line-clamp-2">
											"{citation.quote}"
										</p>
									</div>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}

		<!-- Stats Panel (show when ready) -->
		{#if stats && (stage === 'ready' || isProcessing)}
			<div class="w-full flex flex-col items-center gap-2 mt-4">
				<div class="flex items-center justify-center gap-6 text-white/40 text-xs">
					<div class="flex items-center gap-1.5">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
						</svg>
						<span>{stats.total_files} files</span>
					</div>
					<div class="flex items-center gap-1.5">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7" />
						</svg>
						<span>{stats.total_chunks} chunks</span>
					</div>
					{#if stats.total_size_bytes > 0}
						<div class="flex items-center gap-1.5">
							<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
							</svg>
							<span>{formatSize(stats.total_size_bytes)}</span>
						</div>
					{/if}
				</div>
				<!-- Pipeline timing summary (show when ready) -->
				{#if stage === 'ready' && timing?.phase_times && Object.keys(timing.phase_times).length > 0}
					<div class="flex items-center justify-center gap-1 text-white/30 text-xs">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
						{#each Object.entries(timing.phase_times) as [phase, duration], i}
							{#if i > 0}<span class="text-white/20">+</span>{/if}
							<span class="text-white/40">{getPhaseLabel(phase).toLowerCase()}</span>
							<span class="text-green-400/70 font-mono">{formatTime(duration)}</span>
						{/each}
						{#if finalPipelineTime}
							<span class="text-white/20">=</span>
							<span class="text-blue-400 font-mono">{formatTime(finalPipelineTime)}</span>
						{/if}
					</div>
				{/if}
			</div>
		{/if}

		<!-- Empty state hint -->
		{#if stage === 'idle' && stagedFiles.length === 0}
			<div class="text-center text-white/40 mt-2">
				<p class="text-sm">Upload files to create a searchable knowledge base.</p>
			</div>
		{/if}

		<!-- Debug log (visible on screen) -->
		{#if debugLog.length > 0}
			<div class="w-full mt-4 p-3 bg-black/50 rounded-lg font-mono text-xs text-green-400 max-h-40 overflow-y-auto">
				{#each debugLog as msg}
					<div>{msg}</div>
				{/each}
			</div>
		{/if}

		<!-- Ready state with no query yet -->
		{#if stage === 'ready' && !answerResponse && !searchQuery}
			<div class="text-center text-white/30 mt-4">
				<p class="text-sm">Your documents are ready. Ask anything!</p>
			</div>
		{/if}
	</div>
</main>

<style>
	.line-clamp-2 {
		display: -webkit-box;
		-webkit-line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
</style>
