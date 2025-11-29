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

	// State
	let stage = $state<string>('idle');
	let directoryPath = $state('');
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
			}
		};

		eventSource.onerror = () => {
			console.error('SSE connection error');
		};

		return () => {
			eventSource.close();
			// Clean up query timer
			if (queryTimerInterval) {
				clearInterval(queryTimerInterval);
			}
		};
	});

	// Handlers
	async function handleProcess() {
		if (!directoryPath.trim()) return;

		error = null;
		answerResponse = null;
		finalPipelineTime = null;
		timing = null;

		try {
			const response = await fetch('/api/process', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ path: directoryPath.trim() })
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
				if (directoryPath.trim() && !isProcessing) {
					handleProcess();
				}
			} else if (stage === 'ready') {
				if (searchQuery.trim() && canSearch) {
					handleAsk();
				}
			}
		}
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

		<!-- Path Input (show when idle) -->
		{#if stage === 'idle' || stage === 'error'}
			<div class="w-full mt-4">
				<label class="text-white/60 text-sm mb-2 block">Load your documents</label>
				<div class="flex gap-3">
					<input
						type="text"
						bind:value={directoryPath}
						onkeydown={handleKeydown}
						placeholder="/path/to/your/project"
						disabled={isProcessing}
						class="flex-1 px-5 py-4 text-lg bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all disabled:opacity-50"
					/>
					<button
						onclick={handleProcess}
						disabled={!directoryPath.trim() || isProcessing}
						class="px-6 py-4 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 disabled:cursor-not-allowed text-white font-medium rounded-xl transition-colors"
					>
						Load
					</button>
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
						{#if timing?.phase_elapsed}
							<span class="text-blue-400 text-sm font-mono">
								{formatTime(timing.phase_elapsed)}
							</span>
						{/if}
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

				<!-- Completed phases timing -->
				{#if timing?.phase_times && Object.keys(timing.phase_times).length > 0}
					<div class="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-xs">
						{#each Object.entries(timing.phase_times) as [phase, duration]}
							<span class="text-white/40">
								{getPhaseLabel(phase)}: <span class="text-green-400 font-mono">{formatTime(duration)}</span>
							</span>
						{/each}
					</div>
				{/if}

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
			<div class="w-full flex items-center justify-center gap-6 text-white/40 text-xs mt-4">
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
				{#if finalPipelineTime && stage === 'ready'}
					<div class="flex items-center gap-1.5">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
						<span>processed in <span class="text-blue-400 font-mono">{formatTime(finalPipelineTime)}</span></span>
					</div>
				{/if}
			</div>
		{/if}

		<!-- Empty state hint -->
		{#if stage === 'idle'}
			<div class="text-center text-white/40 mt-6">
				<p class="text-sm">Load a directory to create a searchable knowledge base.</p>
				<p class="mt-2 text-xs text-white/30">Example: <code class="bg-white/10 px-2 py-0.5 rounded">~/Documents/my-project</code></p>
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
