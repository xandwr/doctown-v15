<script lang="ts">
	import { onMount } from 'svelte';
	import { SignedIn, SignedOut, SignInButton, UserButton } from 'svelte-clerk';

	// Types
	interface Citation {
		id: number;
		file_path: string;
		chunk_index: number;
		quote: string;
		start_char: number | null;
		end_char: number | null;
	}

	interface CitationContext {
		file_path: string;
		file_id: number;
		chunk: DocpackChunk;
		full_text: string;
		prev_chunk: DocpackChunk | null;
		next_chunk: DocpackChunk | null;
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

	// Docpack Explorer Types
	interface DocpackFile {
		id: number;
		path: string;
		extension: string | null;
		size_bytes: number;
		is_binary: boolean;
		chunk_count: number;
		preview: string | null;
	}

	interface DocpackChunk {
		id: number;
		chunk_index: number;
		text: string;
		token_count: number;
		start_char: number;
		end_char: number;
		summary: string | null;
		media_type: string | null;
	}

	interface DocpackOverview {
		metadata: Record<string, string | null>;
		files: DocpackFile[];
		stats: Record<string, number>;
	}

	interface DocpackFileDetail {
		file: DocpackFile;
		content: string | null;
		chunks: DocpackChunk[];
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

	// Expandable citations state
	let expandedCitations = $state<Set<number>>(new Set());
	let citationContexts = $state<Map<number, CitationContext>>(new Map());
	let loadingCitations = $state<Set<number>>(new Set());

	// Docpack explorer state
	let showExplorer = $state(false);
	let docpackOverview = $state<DocpackOverview | null>(null);
	let selectedFile = $state<DocpackFileDetail | null>(null);
	let selectedChunk = $state<DocpackChunk | null>(null);
	let loadingOverview = $state(false);
	let loadingFile = $state(false);
	let explorerSearch = $state('');
	let explorerView = $state<'files' | 'chunks' | 'content'>('files');
	let expandedChunks = $state<Set<number>>(new Set());

	// Derived
	let progressPercent = $derived(
		progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0
	);

	let isProcessing = $derived(
		stage === 'freezing' || stage === 'chunking' || stage === 'embedding' || stage === 'summarizing' || stage === 'vision'
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
				// Capture final pipeline time when transitioning to ready or on initial connect to ready state
				// Prefer phase_times sum (accurate) over pipeline_elapsed (wall clock which keeps ticking)
				if (data.stage === 'ready' && !finalPipelineTime) {
					if (data.timing.phase_times && Object.keys(data.timing.phase_times).length > 0) {
						// Use sum of phase times - this is the accurate measure
						const totalFromPhases = Object.values(data.timing.phase_times as Record<string, number>).reduce((sum, t) => sum + t, 0);
						if (totalFromPhases > 0) {
							finalPipelineTime = totalFromPhases;
						}
					}
				}
			}

			// Handle processing timer
			const isNowProcessing = ['freezing', 'chunking', 'embedding', 'summarizing', 'vision'].includes(data.stage);
			const wasProcessing = ['freezing', 'chunking', 'embedding', 'summarizing', 'vision'].includes(prevStage);
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

	async function handleUnload() {
		try {
			const response = await fetch('/api/unload', {
				method: 'POST'
			});

			if (!response.ok) {
				const data = await response.json();
				error = data.detail || 'Failed to unload';
			} else {
				// Reset local state
				answerResponse = null;
				searchQuery = '';
				finalPipelineTime = null;
				timing = null;
				stats = null;
				excludedDirs = new Map();
				// SSE will update stage to 'idle'
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Network error';
		}
	}

	// Directories to exclude from uploads (matches docpack/ingest/sources.py IGNORE_DIRS)
	const IGNORE_DIRS = new Set([
		'.git', '.svn', '.hg', '.bzr',
		'__pycache__', 'node_modules',
		'.tox', '.nox', '.mypy_cache', '.pytest_cache', '.ruff_cache',
		'.coverage', 'htmlcov',
		'dist', 'build', '.next', '.nuxt', '.output',
		'.venv', 'venv', '.env', 'env',
		'target',  // Rust/Cargo
		'.gradle', '.idea', '.vscode',
		'vendor',  // Go, PHP
		'Pods',    // iOS
	]);

	// File patterns to exclude
	const IGNORE_FILES = new Set(['.DS_Store', 'Thumbs.db', '.gitignore', '.gitattributes']);
	const IGNORE_EXTENSIONS = new Set(['.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe', '.o', '.a', '.lib']);

	// Size threshold for auto-excluding dense directories (50MB)
	const DENSE_DIR_THRESHOLD = 50 * 1024 * 1024;

	interface FilterResult {
		files: File[];
		excluded: Map<string, { count: number; size: number; reason: string }>;
	}

	function filterFiles(files: FileList | File[]): FilterResult {
		const filtered: File[] = [];
		const excluded = new Map<string, { count: number; size: number; reason: string }>();
		const dirSizes = new Map<string, number>();

		// First pass: calculate directory sizes
		for (let i = 0; i < files.length; i++) {
			const file = files[i] as File & { webkitRelativePath?: string };
			const path = file.webkitRelativePath || file.name;
			const parts = path.split('/');

			// Accumulate size for each directory level
			for (let j = 1; j < parts.length; j++) {
				const dirPath = parts.slice(0, j).join('/');
				dirSizes.set(dirPath, (dirSizes.get(dirPath) || 0) + file.size);
			}
		}

		// Find dense directories (over threshold)
		const denseDirs = new Set<string>();
		for (const [dir, size] of dirSizes) {
			// Only flag as dense if it's a single component (top-level) or immediate child
			const depth = dir.split('/').length;
			if (depth <= 2 && size > DENSE_DIR_THRESHOLD) {
				// Check if this isn't already covered by IGNORE_DIRS
				const dirName = dir.split('/').pop() || '';
				if (!IGNORE_DIRS.has(dirName)) {
					denseDirs.add(dir);
				}
			}
		}

		// Second pass: filter files
		for (let i = 0; i < files.length; i++) {
			const file = files[i] as File & { webkitRelativePath?: string };
			const path = file.webkitRelativePath || file.name;
			const parts = path.split('/');
			const fileName = parts[parts.length - 1];
			const ext = fileName.includes('.') ? '.' + fileName.split('.').pop()?.toLowerCase() : '';

			// Check if file is in an ignored directory
			let excludeReason: string | null = null;
			let excludeDir: string | null = null;

			for (let j = 0; j < parts.length - 1; j++) {
				const dirName = parts[j];
				const dirPath = parts.slice(0, j + 1).join('/');

				if (IGNORE_DIRS.has(dirName)) {
					excludeReason = 'ignored directory';
					excludeDir = dirPath;
					break;
				}

				if (denseDirs.has(dirPath)) {
					excludeReason = 'large directory';
					excludeDir = dirPath;
					break;
				}
			}

			// Check file-level ignores
			if (!excludeReason) {
				if (IGNORE_FILES.has(fileName)) {
					excludeReason = 'ignored file';
					excludeDir = fileName;
				} else if (IGNORE_EXTENSIONS.has(ext)) {
					excludeReason = 'binary/compiled';
					excludeDir = `*${ext}`;
				}
			}

			if (excludeReason && excludeDir) {
				const existing = excluded.get(excludeDir) || { count: 0, size: 0, reason: excludeReason };
				existing.count++;
				existing.size += file.size;
				excluded.set(excludeDir, existing);
			} else {
				filtered.push(file);
			}
		}

		return { files: filtered, excluded };
	}

	// State for showing excluded files
	let excludedDirs = $state<Map<string, { count: number; size: number; reason: string }>>(new Map());

	// File upload handlers
	async function uploadFiles(files: FileList | File[]) {
		if (!files || files.length === 0) return;

		isUploading = true;
		error = null;

		try {
			log('uploadFiles started');

			// Filter out ignored directories and files
			const { files: filteredFiles, excluded } = filterFiles(files);
			excludedDirs = excluded;

			if (excluded.size > 0) {
				const totalExcluded = Array.from(excluded.values()).reduce((sum, e) => sum + e.count, 0);
				log(`Filtered out ${totalExcluded} files from ${excluded.size} directories`);
			}

			if (filteredFiles.length === 0) {
				error = 'No files to upload after filtering. All files were in ignored directories.';
				return;
			}

			const formData = new FormData();
			let totalSize = 0;
			for (const file of filteredFiles) {
				const f = file as File & { webkitRelativePath?: string };
				// Use webkitRelativePath if available (folder upload), otherwise just the name
				const uploadPath = f.webkitRelativePath || f.name;
				log(`append: ${uploadPath}`);
				formData.append('files', file, uploadPath);
				totalSize += file.size;
			}
			log(`sending ${filteredFiles.length} files (${totalSize} bytes)`);

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
				excludedDirs = new Map();
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
				// Shift+Enter adds newline, Enter alone submits
				if (!e.shiftKey && searchQuery.trim() && canSearch) {
					e.preventDefault();
					handleAsk();
				}
			}
		}
	}

	function autoResizeTextarea(e: Event) {
		const textarea = e.target as HTMLTextAreaElement;
		textarea.style.height = 'auto';
		textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
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
			case 'vision': return 'Analyzing images';
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

	// Citation expansion handlers
	async function toggleCitation(citation: Citation) {
		const key = citation.id;
		if (expandedCitations.has(key)) {
			// Collapse
			expandedCitations = new Set([...expandedCitations].filter(k => k !== key));
		} else {
			// Expand - fetch context if not already loaded
			if (!citationContexts.has(key)) {
				loadingCitations = new Set([...loadingCitations, key]);
				try {
					const response = await fetch(`/api/citation/${encodeURIComponent(citation.file_path)}/${citation.chunk_index}`);
					if (response.ok) {
						const context: CitationContext = await response.json();
						citationContexts = new Map([...citationContexts, [key, context]]);
					}
				} catch (e) {
					console.error('Failed to load citation context:', e);
				} finally {
					loadingCitations = new Set([...loadingCitations].filter(k => k !== key));
				}
			}
			expandedCitations = new Set([...expandedCitations, key]);
		}
	}

	// Docpack explorer handlers
	async function openExplorer() {
		showExplorer = true;
		if (!docpackOverview) {
			await loadDocpackOverview();
		}
	}

	function closeExplorer() {
		showExplorer = false;
		selectedFile = null;
		selectedChunk = null;
		explorerView = 'files';
	}

	async function loadDocpackOverview() {
		loadingOverview = true;
		try {
			const response = await fetch('/api/docpack/overview');
			if (response.ok) {
				docpackOverview = await response.json();
			}
		} catch (e) {
			console.error('Failed to load docpack overview:', e);
		} finally {
			loadingOverview = false;
		}
	}

	async function selectFile(file: DocpackFile) {
		loadingFile = true;
		explorerView = 'chunks';
		selectedChunk = null;
		try {
			const response = await fetch(`/api/docpack/file/${file.id}`);
			if (response.ok) {
				selectedFile = await response.json();
			}
		} catch (e) {
			console.error('Failed to load file detail:', e);
		} finally {
			loadingFile = false;
		}
	}

	function selectChunk(chunk: DocpackChunk) {
		selectedChunk = chunk;
		explorerView = 'content';
	}

	function backToFiles() {
		selectedFile = null;
		selectedChunk = null;
		explorerView = 'files';
	}

	function backToChunks() {
		selectedChunk = null;
		explorerView = 'chunks';
	}

	function toggleChunkExpand(chunkId: number) {
		if (expandedChunks.has(chunkId)) {
			expandedChunks = new Set([...expandedChunks].filter(id => id !== chunkId));
		} else {
			expandedChunks = new Set([...expandedChunks, chunkId]);
		}
	}

	// Filtered files for explorer search
	let filteredFiles = $derived(
		docpackOverview?.files.filter(f =>
			f.path.toLowerCase().includes(explorerSearch.toLowerCase())
		) ?? []
	);

	function getExtensionColor(ext: string | null): string {
		if (!ext) return 'bg-gray-500/20 text-gray-400';
		const codeExts = ['js', 'ts', 'jsx', 'tsx', 'py', 'rb', 'go', 'rs', 'java', 'c', 'cpp', 'cs'];
		const docExts = ['md', 'txt', 'rst'];
		const dataExts = ['json', 'yaml', 'yml', 'xml', 'csv', 'toml'];
		if (codeExts.includes(ext)) return 'bg-blue-500/20 text-blue-400';
		if (docExts.includes(ext)) return 'bg-green-500/20 text-green-400';
		if (dataExts.includes(ext)) return 'bg-yellow-500/20 text-yellow-400';
		if (ext === 'pdf') return 'bg-red-500/20 text-red-400';
		return 'bg-purple-500/20 text-purple-400';
	}
</script>

<main class="min-h-screen bg-slate-900 flex flex-col items-center px-3 py-6 sm:px-4 sm:py-12">
	<!-- Auth Header -->
	<div class="w-full max-w-3xl flex justify-end mb-4">
		<SignedOut>
			<SignInButton mode="modal">
				<button class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
					Sign In
				</button>
			</SignInButton>
		</SignedOut>
		<SignedIn>
			<UserButton />
		</SignedIn>
	</div>

	<div class="w-full max-w-3xl flex flex-col items-center gap-4 sm:gap-6">
		<!-- Logo/Title -->
		<h1 class="text-3xl sm:text-4xl font-bold text-white tracking-tight">
			Doctown
		</h1>
		<p class="text-white/60 text-sm sm:text-base -mt-2 sm:-mt-3">
			Instant answers from your documents
		</p>

		<!-- Error Display -->
		{#if error}
			<div class="w-full bg-red-500/20 border border-red-500/50 rounded-xl p-3 sm:p-4 text-red-200 text-xs sm:text-sm">
				{error}
			</div>
		{/if}

		<!-- File Upload UI (show when idle) -->
		{#if stage === 'idle' || stage === 'error'}
			<div class="w-full mt-2 sm:mt-4 space-y-3 sm:space-y-4">
				<!-- Drop zone -->
				<div
					class="relative border-2 border-dashed rounded-xl sm:rounded-2xl p-6 sm:p-8 text-center transition-all cursor-pointer
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
						<div class="flex flex-col items-center gap-2 sm:gap-3">
							<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 sm:h-10 sm:w-10 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
							</svg>
							<div>
								<p class="text-white/70 text-base sm:text-lg">Drop files here or tap to browse</p>
								<p class="text-white/40 text-xs sm:text-sm mt-1">Upload documents, code, or any text files</p>
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
					<div class="bg-white/5 rounded-xl p-3 sm:p-4 space-y-2">
						<div class="flex items-center justify-between mb-2 sm:mb-3">
							<span class="text-white/60 text-xs sm:text-sm">
								{stagedFiles.length} file{stagedFiles.length !== 1 ? 's' : ''} staged ({formatSize(stagedTotalSize)})
							</span>
							<button
								onclick={clearAllFiles}
								class="text-red-400/70 hover:text-red-400 text-xs transition-colors"
							>
								Clear all
							</button>
						</div>

						<div class="max-h-40 sm:max-h-48 overflow-y-auto space-y-1">
							{#each stagedFiles as file}
								<div class="flex items-center gap-2 sm:gap-3 py-1.5 px-2 rounded-lg hover:bg-white/5 group">
									<!-- File icon -->
									<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-white/40 shrink-0 hidden sm:block" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
									<span class="flex-1 text-white/70 text-xs sm:text-sm truncate" title={file.name}>
										{file.name}
									</span>

									<!-- File size -->
									<span class="text-white/30 text-xs shrink-0 hidden sm:inline">
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
							class="w-full mt-2 sm:mt-3 px-4 sm:px-6 py-2.5 sm:py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 disabled:cursor-not-allowed text-white text-sm sm:text-base font-medium rounded-xl transition-colors"
						>
							Process {stagedFiles.length} file{stagedFiles.length !== 1 ? 's' : ''}
						</button>
					</div>
				{/if}

				<!-- Excluded directories notice -->
				{#if excludedDirs.size > 0}
					{@const totalExcluded = Array.from(excludedDirs.values()).reduce((sum, e) => sum + e.count, 0)}
					{@const totalSize = Array.from(excludedDirs.values()).reduce((sum, e) => sum + e.size, 0)}
					<div class="bg-yellow-500/10 border border-yellow-500/20 rounded-xl p-3 sm:p-4">
						<div class="flex items-start gap-2">
							<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-yellow-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
							</svg>
							<div class="flex-1 min-w-0">
								<p class="text-yellow-200/90 text-sm font-medium">
									Excluded {totalExcluded} files ({formatSize(totalSize)})
								</p>
								<div class="mt-2 flex flex-wrap gap-1.5">
									{#each [...excludedDirs.entries()].sort((a, b) => b[1].size - a[1].size).slice(0, 6) as [dir, info]}
										<span class="inline-flex items-center gap-1 px-2 py-0.5 bg-yellow-500/10 rounded text-xs text-yellow-300/70" title="{info.count} files, {formatSize(info.size)} - {info.reason}">
											<span class="truncate max-w-24">{dir}</span>
											<span class="text-yellow-400/50">({formatSize(info.size)})</span>
										</span>
									{/each}
									{#if excludedDirs.size > 6}
										<span class="text-yellow-400/50 text-xs">+{excludedDirs.size - 6} more</span>
									{/if}
								</div>
							</div>
						</div>
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
			<div class="w-full bg-white/5 rounded-xl p-4 sm:p-6 mt-2 sm:mt-4">
				<div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 sm:gap-0 mb-3">
					<span class="text-white/80 font-medium text-sm sm:text-base">
						{getPhaseLabel(stage)}...
					</span>
					<div class="flex items-center gap-2 sm:gap-3">
						<span class="text-blue-400 text-xs sm:text-sm font-mono tabular-nums">
							{formatTime(localPhaseElapsed)}
						</span>
						<span class="text-white/60 text-xs sm:text-sm">
							{progress.current} / {progress.total}
						</span>
					</div>
				</div>
				<div class="w-full bg-white/10 rounded-full h-1.5 sm:h-2 overflow-hidden">
					<div
						class="h-full bg-blue-500 transition-all duration-300 ease-out"
						style="width: {progressPercent}%"
					></div>
				</div>

				<!-- Completed phases timing + total elapsed -->
				<div class="flex flex-wrap items-center gap-x-3 sm:gap-x-4 gap-y-1 mt-2 sm:mt-3 text-xs">
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

				<p class="text-white/40 text-xs sm:text-sm mt-2">
					{#if stage === 'vision'}
						Analyzing images with vision model...
					{:else if stage === 'summarizing'}
						Generating AI summaries for better answers...
					{:else if stage === 'embedding'}
						Creating semantic vectors...
					{:else}
						Processing your files...
					{/if}
				</p>
			</div>
		{/if}

		<!-- Search Bar (show when ready) - OpenAI-style auto-expanding -->
		{#if stage === 'ready'}
			<div class="w-full mt-2 sm:mt-4">
				<div class="relative">
					<div class="absolute left-3 sm:left-5 top-4 sm:top-5 text-white/40">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
						</svg>
					</div>
					<textarea
						bind:value={searchQuery}
						onkeydown={handleKeydown}
						oninput={autoResizeTextarea}
						placeholder="Ask anything about your documents..."
						disabled={isAsking}
						rows="1"
						class="w-full pl-10 sm:pl-14 pr-12 sm:pr-16 py-3.5 sm:py-5 text-base sm:text-lg bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl sm:rounded-2xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none overflow-hidden"
						style="min-height: 52px; max-height: 200px;"
					></textarea>
					{#if isAsking}
						<div class="absolute right-3 sm:right-5 top-4 sm:top-5 flex items-center gap-2">
							<span class="text-blue-400 text-xs sm:text-sm font-mono tabular-nums">
								{queryElapsed !== null ? formatTime(queryElapsed) : ''}
							</span>
							<svg class="h-4 w-4 sm:h-5 sm:w-5 animate-spin text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						</div>
					{:else}
						<button
							onclick={handleAsk}
							disabled={!searchQuery.trim() || !canSearch}
							aria-label="Submit search"
							class="absolute right-3 sm:right-5 top-3.5 sm:top-4 p-1.5 sm:p-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-white/10 disabled:text-white/30 text-white transition-colors"
						>
							<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 sm:h-5 sm:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
							</svg>
						</button>
					{/if}
				</div>
				<p class="text-white/30 text-xs mt-2 text-center">Press Enter to search, Shift+Enter for new line</p>
			</div>
		{/if}

		<!-- Answer Display -->
		{#if answerResponse}
			<div class="w-full mt-2 space-y-3 sm:space-y-4">
				<!-- Main Answer Card -->
				<div class="bg-white/8 border border-white/10 rounded-xl sm:rounded-2xl p-4 sm:p-6">
					<!-- Answer text -->
					<p class="text-white text-base sm:text-lg leading-relaxed">
						{answerResponse.answer}
					</p>

					<!-- Confidence indicator -->
					<div class="mt-3 sm:mt-4 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs sm:text-sm">
						<span class="{getConfidenceColor(answerResponse.confidence)} font-mono">
							{getConfidenceIcon(answerResponse.confidence)}
						</span>
						<span class="text-white/50">
							{answerResponse.confidence} confidence
						</span>
						<span class="text-white/30">|</span>
						<span class="text-white/40">
							{answerResponse.sources_used}/{answerResponse.sources_retrieved} sources
						</span>
						{#if queryElapsed !== null}
							<span class="text-white/30">|</span>
							<span class="text-white/40">
								<span class="text-blue-400 font-mono">{formatTime(queryElapsed)}</span>
							</span>
						{/if}
					</div>
				</div>

				<!-- Citations (Expandable) -->
				{#if answerResponse.citations.length > 0}
					<div class="space-y-2">
						<h3 class="text-white/50 text-xs font-medium uppercase tracking-wide px-1">Sources (click to expand)</h3>
						{#each answerResponse.citations as citation}
							{@const isExpanded = expandedCitations.has(citation.id)}
							{@const isLoading = loadingCitations.has(citation.id)}
							{@const context = citationContexts.get(citation.id)}
							<div class="bg-white/4 border border-white/8 rounded-lg sm:rounded-xl overflow-hidden transition-all">
								<!-- Collapsed header (always visible, clickable) -->
								<button
									onclick={() => toggleCitation(citation)}
									class="w-full p-3 sm:p-4 hover:bg-white/6 transition-colors text-left"
								>
									<div class="flex items-start gap-2 sm:gap-3">
										<!-- Citation number -->
										<span class="text-blue-400 font-mono text-xs sm:text-sm font-medium shrink-0">
											[{citation.id}]
										</span>
										<div class="flex-1 min-w-0">
											<!-- File path -->
											<div class="flex flex-col sm:flex-row sm:items-center gap-0.5 sm:gap-2 mb-1.5 sm:mb-2">
												<span class="text-blue-300 text-xs sm:text-sm font-medium truncate">
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
											<!-- Quote preview -->
											{#if !isExpanded}
												<p class="text-white/60 text-xs sm:text-sm leading-relaxed line-clamp-2">
													"{citation.quote}"
												</p>
											{/if}
										</div>
										<!-- Expand indicator -->
										<div class="shrink-0 text-white/30">
											{#if isLoading}
												<svg class="h-4 w-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
													<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
													<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
												</svg>
											{:else}
												<svg class="h-4 w-4 transition-transform {isExpanded ? 'rotate-180' : ''}" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
													<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
												</svg>
											{/if}
										</div>
									</div>
								</button>

								<!-- Expanded content -->
								{#if isExpanded && context}
									<div class="border-t border-white/10 bg-black/20">
										<!-- Context navigation -->
										{#if context.prev_chunk || context.next_chunk}
											<div class="flex items-center justify-between px-4 py-2 border-b border-white/5 text-xs">
												<span class="text-white/40">
													Chunk {context.chunk.chunk_index + 1} of file
												</span>
												<div class="flex gap-2">
													{#if context.prev_chunk}
														<span class="text-white/30">← prev</span>
													{/if}
													{#if context.next_chunk}
														<span class="text-white/30">next →</span>
													{/if}
												</div>
											</div>
										{/if}

										<!-- Previous chunk (faded) -->
										{#if context.prev_chunk}
											<div class="px-4 py-3 bg-white/2 border-b border-white/5">
												<div class="text-white/25 text-xs mb-1 font-mono">↑ Previous chunk</div>
												<p class="text-white/30 text-xs sm:text-sm leading-relaxed line-clamp-3 font-mono">
													{context.prev_chunk.text}
												</p>
											</div>
										{/if}

										<!-- Main chunk (highlighted) -->
										<div class="px-4 py-4 bg-blue-500/5 border-l-2 border-blue-400">
											<div class="flex items-center gap-2 mb-2">
												<span class="text-blue-400 text-xs font-medium">Source content</span>
												{#if context.chunk.summary}
													<span class="text-white/30 text-xs">• {context.chunk.token_count} tokens</span>
												{/if}
											</div>
											<p class="text-white/80 text-sm sm:text-base leading-relaxed font-mono whitespace-pre-wrap">
												{context.full_text}
											</p>
											{#if context.chunk.summary}
												<div class="mt-3 pt-3 border-t border-white/10">
													<span class="text-white/40 text-xs">Summary: </span>
													<span class="text-white/60 text-xs">{context.chunk.summary}</span>
												</div>
											{/if}
										</div>

										<!-- Next chunk (faded) -->
										{#if context.next_chunk}
											<div class="px-4 py-3 bg-white/2 border-t border-white/5">
												<div class="text-white/25 text-xs mb-1 font-mono">↓ Next chunk</div>
												<p class="text-white/30 text-xs sm:text-sm leading-relaxed line-clamp-3 font-mono">
													{context.next_chunk.text}
												</p>
											</div>
										{/if}

										<!-- Actions -->
										<div class="px-4 py-2 border-t border-white/10 flex items-center justify-between">
											<span class="text-white/30 text-xs">
												chars {context.chunk.start_char}-{context.chunk.end_char}
											</span>
											<button
												onclick={() => { selectFile({ id: context.file_id, path: context.file_path, extension: null, size_bytes: 0, is_binary: false, chunk_count: 0, preview: null }); openExplorer(); }}
												class="text-blue-400/70 hover:text-blue-400 text-xs transition-colors"
											>
												View full file →
											</button>
										</div>
									</div>
								{/if}
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}

		<!-- Stats Panel (show when ready) -->
		{#if stats && (stage === 'ready' || isProcessing)}
			<div class="w-full flex flex-col items-center gap-1.5 sm:gap-2 mt-2 sm:mt-4">
				<!-- Explore button (prominent when ready) -->
				{#if stage === 'ready'}
					<button
						onclick={openExplorer}
						class="group flex items-center gap-2 px-4 py-2 bg-linear-to-r from-blue-600/20 to-purple-600/20 hover:from-blue-600/30 hover:to-purple-600/30 border border-white/10 hover:border-white/20 rounded-xl transition-all"
					>
						<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
						</svg>
						<span class="text-white/80 text-sm font-medium">Explore Documents</span>
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 text-white/40 group-hover:text-white/60 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
						</svg>
					</button>
				{/if}

				<div class="flex items-center justify-center gap-3 sm:gap-6 text-white/40 text-xs">
					<div class="flex items-center gap-1 sm:gap-1.5">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 sm:h-3.5 sm:w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
						</svg>
						<span>{stats.total_files} files</span>
					</div>
					<div class="flex items-center gap-1 sm:gap-1.5">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 sm:h-3.5 sm:w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7" />
						</svg>
						<span>{stats.total_chunks} chunks</span>
					</div>
					{#if stats.total_size_bytes > 0}
						<div class="flex items-center gap-1 sm:gap-1.5">
							<svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 sm:h-3.5 sm:w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
							</svg>
							<span>{formatSize(stats.total_size_bytes)}</span>
						</div>
					{/if}
				</div>
				<!-- Pipeline timing summary (show when ready) -->
				{#if stage === 'ready' && timing?.phase_times && Object.keys(timing.phase_times).length > 0}
					<div class="flex flex-wrap items-center justify-center gap-1 text-white/30 text-xs px-2">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3 sm:h-3.5 sm:w-3.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
						</svg>
						{#each Object.entries(timing.phase_times) as [phase, duration], i}
							{#if i > 0}<span class="text-white/20">+</span>{/if}
							<span class="text-white/40 hidden sm:inline">{getPhaseLabel(phase).toLowerCase()}</span>
							<span class="text-green-400/70 font-mono">{formatTime(duration)}</span>
						{/each}
						{#if finalPipelineTime}
							<span class="text-white/20">=</span>
							<span class="text-blue-400 font-mono">{formatTime(finalPipelineTime)}</span>
						{/if}
					</div>
				{/if}
				<!-- Unload button (show when ready) -->
				{#if stage === 'ready'}
					<button
						onclick={handleUnload}
						class="mt-2 px-3 py-1 text-xs text-white/40 hover:text-white/70 hover:bg-white/5 rounded-lg transition-colors"
					>
						Clear & start over
					</button>
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

	<!-- Docpack Explorer Modal -->
	{#if showExplorer}
		<div class="fixed inset-0 z-50 flex items-center justify-center p-4">
			<!-- Backdrop -->
			<div
				class="absolute inset-0 bg-black/80 backdrop-blur-sm"
				onclick={closeExplorer}
				role="button"
				tabindex="-1"
				onkeydown={(e) => e.key === 'Escape' && closeExplorer()}
			></div>

			<!-- Modal -->
			<div class="relative w-full max-w-5xl h-[85vh] bg-slate-900 border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
				<!-- Header -->
				<div class="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-slate-800/50">
					<div class="flex items-center gap-4">
						<div class="flex items-center gap-2">
							<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
							</svg>
							<h2 class="text-lg font-semibold text-white">Document Explorer</h2>
						</div>
						<!-- Breadcrumb -->
						<div class="flex items-center gap-2 text-sm">
							<button
								onclick={backToFiles}
								class="text-white/50 hover:text-white/80 transition-colors {explorerView === 'files' ? 'text-blue-400' : ''}"
							>
								Files
							</button>
							{#if selectedFile}
								<span class="text-white/30">/</span>
								<button
									onclick={backToChunks}
									class="text-white/50 hover:text-white/80 transition-colors truncate max-w-48 {explorerView === 'chunks' ? 'text-blue-400' : ''}"
								>
									{selectedFile.file.path}
								</button>
							{/if}
							{#if selectedChunk}
								<span class="text-white/30">/</span>
								<span class="text-blue-400">Chunk {selectedChunk.chunk_index + 1}</span>
							{/if}
						</div>
					</div>
					<button
						onclick={closeExplorer}
						aria-label="Close explorer"
						class="p-2 text-white/40 hover:text-white/80 hover:bg-white/10 rounded-lg transition-colors"
					>
						<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
						</svg>
					</button>
				</div>

				<!-- Content -->
				<div class="flex-1 overflow-hidden flex">
					{#if loadingOverview}
						<div class="flex-1 flex items-center justify-center">
							<svg class="h-8 w-8 animate-spin text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						</div>
					{:else if explorerView === 'files' && docpackOverview}
						<!-- Files List View -->
						<div class="flex-1 flex flex-col">
							<!-- Search bar -->
							<div class="p-4 border-b border-white/5">
								<div class="relative">
									<svg xmlns="http://www.w3.org/2000/svg" class="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
									</svg>
									<input
										type="text"
										bind:value={explorerSearch}
										placeholder="Filter files..."
										class="w-full pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-white/40 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
									/>
								</div>
							</div>

							<!-- Stats summary -->
							<div class="px-4 py-3 bg-white/2 border-b border-white/5 flex items-center gap-6 text-xs text-white/50">
								<span>{docpackOverview.stats.total_files} files</span>
								<span>{docpackOverview.stats.total_chunks} chunks</span>
								<span>{docpackOverview.stats.total_vectors} vectors</span>
								{#if docpackOverview.stats.total_images > 0}
									<span>{docpackOverview.stats.total_images} images</span>
								{/if}
							</div>

							<!-- File list -->
							<div class="flex-1 overflow-y-auto">
								{#each filteredFiles as file}
									<button
										onclick={() => selectFile(file)}
										class="w-full flex items-center gap-4 px-4 py-3 hover:bg-white/5 border-b border-white/5 transition-colors text-left group"
									>
										<!-- Extension badge -->
										<span class="shrink-0 px-2 py-0.5 rounded text-xs font-mono {getExtensionColor(file.extension)}">
											{file.extension || 'file'}
										</span>

										<!-- File info -->
										<div class="flex-1 min-w-0">
											<div class="text-white/80 text-sm truncate group-hover:text-white">
												{file.path}
											</div>
											{#if file.preview}
												{@const firstLine = file.preview.split('\n').find(line => line.trim().length > 0)?.trim().slice(0, 80)}
												{#if firstLine}
													<div class="text-white/40 text-xs truncate mt-0.5 font-mono">
														{firstLine}{firstLine.length >= 80 ? '...' : ''}
													</div>
												{/if}
											{/if}
										</div>

										<!-- Metadata -->
										<div class="shrink-0 text-right">
											<div class="text-white/50 text-xs">
												{file.chunk_count} chunks
											</div>
											<div class="text-white/30 text-xs">
												{formatSize(file.size_bytes)}
											</div>
										</div>

										<!-- Arrow -->
										<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-white/20 group-hover:text-white/50 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
										</svg>
									</button>
								{/each}
							</div>
						</div>

					{:else if explorerView === 'chunks' && selectedFile}
						<!-- Chunks View -->
						<div class="flex-1 flex flex-col">
							{#if loadingFile}
								<div class="flex-1 flex items-center justify-center">
									<svg class="h-8 w-8 animate-spin text-blue-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
										<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
										<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
									</svg>
								</div>
							{:else}
								<!-- File info header -->
								<div class="px-4 py-3 bg-white/2 border-b border-white/5">
									<div class="flex items-center gap-3">
										<button
											onclick={backToFiles}
											class="shrink-0 p-1.5 -ml-1 text-white/40 hover:text-white/80 hover:bg-white/10 rounded-lg transition-colors"
											aria-label="Back to files"
										>
											<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
											</svg>
										</button>
										<span class="px-2 py-0.5 rounded text-xs font-mono {getExtensionColor(selectedFile.file.extension)}">
											{selectedFile.file.extension || 'file'}
										</span>
										<span class="text-white/60 text-sm">{selectedFile.file.path}</span>
										<span class="text-white/30 text-xs ml-auto">{formatSize(selectedFile.file.size_bytes)}</span>
									</div>
								</div>

								<!-- Chunks list -->
								<div class="flex-1 overflow-y-auto">
									{#each selectedFile.chunks as chunk, i}
										{@const isChunkExpanded = expandedChunks.has(chunk.id)}
										<div class="border-b border-white/5">
											<!-- Chunk header (always visible, clickable to expand) -->
											<button
												onclick={() => toggleChunkExpand(chunk.id)}
												class="w-full flex items-start gap-3 px-4 py-3 hover:bg-white/5 transition-colors text-left group"
											>
												<!-- Chunk index -->
												<div class="shrink-0">
													<span class="inline-block px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs font-mono">
														#{i + 1}
													</span>
												</div>

												<!-- Chunk preview/header -->
												<div class="flex-1 min-w-0">
													<div class="flex items-center gap-2 flex-wrap">
														<span class="text-white/40 text-xs">{chunk.token_count} tokens</span>
														<span class="text-white/30 text-xs">•</span>
														<span class="text-white/30 text-xs">chars {chunk.start_char}-{chunk.end_char}</span>
														{#if chunk.media_type && chunk.media_type !== 'text'}
															<span class="px-1.5 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">
																{chunk.media_type}
															</span>
														{/if}
													</div>
													{#if !isChunkExpanded}
														{#if chunk.summary}
															<!-- Show AI summary when available -->
															<p class="text-white/70 text-sm leading-relaxed mt-1.5 group-hover:text-white/90">
																{chunk.summary}
															</p>
														{:else}
															<!-- Fallback: show first line or clean preview -->
															<p class="text-white/50 text-sm leading-relaxed mt-1.5 line-clamp-1 group-hover:text-white/70 font-mono">
																{chunk.text.split('\n').find(line => line.trim().length > 0)?.trim().slice(0, 100) || chunk.text.slice(0, 100)}...
															</p>
														{/if}
													{/if}
												</div>

												<!-- Expand indicator -->
												<div class="shrink-0 text-white/30">
													<svg class="h-4 w-4 transition-transform {isChunkExpanded ? 'rotate-180' : ''}" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
														<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
													</svg>
												</div>
											</button>

											<!-- Expanded content -->
											{#if isChunkExpanded}
												<div class="px-4 pb-4 bg-white/2">
													<!-- Full text with proper wrapping -->
													<div class="pl-10">
														<pre class="text-white/80 text-sm leading-relaxed whitespace-pre-wrap wrap-break-word font-mono bg-black/20 rounded-lg p-4 max-h-96 overflow-y-auto">{chunk.text}</pre>

														{#if chunk.summary}
															<div class="mt-3 p-3 bg-blue-500/5 border border-blue-500/10 rounded-lg">
																<span class="text-blue-400/70 text-xs font-medium">AI Summary</span>
																<p class="text-white/60 text-sm mt-1 italic">{chunk.summary}</p>
															</div>
														{/if}

														<!-- Actions -->
														<div class="mt-3 flex items-center gap-3">
															<button
																onclick={(e) => { e.stopPropagation(); selectChunk(chunk); }}
																class="text-blue-400/70 hover:text-blue-400 text-xs transition-colors flex items-center gap-1"
															>
																<svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
																	<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
																	<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
																</svg>
																Open full view
															</button>
														</div>
													</div>
												</div>
											{/if}
										</div>
									{/each}
								</div>
							{/if}
						</div>

					{:else if explorerView === 'content' && selectedChunk && selectedFile}
						<!-- Full Content View -->
						<div class="flex-1 flex flex-col">
							<!-- Chunk header -->
							<div class="px-4 py-3 bg-blue-500/10 border-b border-blue-500/20 flex items-center justify-between">
								<div class="flex items-center gap-3">
									<span class="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs font-mono">
										Chunk #{selectedChunk.chunk_index + 1}
									</span>
									<span class="text-white/50 text-xs">
										{selectedChunk.token_count} tokens • chars {selectedChunk.start_char}-{selectedChunk.end_char}
									</span>
								</div>
								{#if selectedChunk.media_type && selectedChunk.media_type !== 'text'}
									<span class="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs">
										{selectedChunk.media_type}
									</span>
								{/if}
							</div>

							<!-- Full text content -->
							<div class="flex-1 overflow-y-auto p-6">
								<pre class="text-white/80 text-sm leading-relaxed whitespace-pre-wrap font-mono">{selectedChunk.text}</pre>

								{#if selectedChunk.summary}
									<div class="mt-6 pt-4 border-t border-white/10">
										<h4 class="text-white/50 text-xs uppercase tracking-wide mb-2">AI Summary</h4>
										<p class="text-white/60 text-sm italic">{selectedChunk.summary}</p>
									</div>
								{/if}
							</div>

							<!-- Navigation -->
							<div class="px-4 py-3 border-t border-white/10 flex items-center justify-between bg-white/2">
								<button
									onclick={() => {
										const prevIdx = selectedChunk!.chunk_index - 1;
										const prevChunk = selectedFile?.chunks.find(c => c.chunk_index === prevIdx);
										if (prevChunk) selectChunk(prevChunk);
									}}
									disabled={selectedChunk.chunk_index === 0}
									class="flex items-center gap-2 px-3 py-1.5 text-sm text-white/60 hover:text-white/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
								>
									<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
									</svg>
									Previous chunk
								</button>

								<span class="text-white/40 text-xs">
									{selectedChunk.chunk_index + 1} of {selectedFile.chunks.length}
								</span>

								<button
									onclick={() => {
										const nextIdx = selectedChunk!.chunk_index + 1;
										const nextChunk = selectedFile?.chunks.find(c => c.chunk_index === nextIdx);
										if (nextChunk) selectChunk(nextChunk);
									}}
									disabled={selectedChunk.chunk_index >= selectedFile.chunks.length - 1}
									class="flex items-center gap-2 px-3 py-1.5 text-sm text-white/60 hover:text-white/90 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
								>
									Next chunk
									<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
									</svg>
								</button>
							</div>
						</div>
					{/if}
				</div>

				<!-- Footer with metadata -->
				{#if docpackOverview?.metadata}
					<div class="px-6 py-3 border-t border-white/10 bg-slate-800/30 flex items-center justify-between text-xs text-white/40">
						<div class="flex items-center gap-4">
							{#if docpackOverview.metadata.created_at}
								<span>Created: {new Date(docpackOverview.metadata.created_at).toLocaleString()}</span>
							{/if}
							{#if docpackOverview.metadata.docpack_version}
								<span>v{docpackOverview.metadata.docpack_version}</span>
							{/if}
						</div>
						<div class="flex items-center gap-2">
							<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
							</svg>
							<span>SQLite-powered semantic search</span>
						</div>
					</div>
				{/if}
			</div>
		</div>
	{/if}
</main>

<style>
	.line-clamp-2 {
		display: -webkit-box;
		line-clamp: 2;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
	.line-clamp-3 {
		display: -webkit-box;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
</style>
