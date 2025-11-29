<script lang="ts">
	import { onMount } from 'svelte';

	// Types
	interface SearchResult {
		file_path: string;
		chunk_index: number;
		text: string;
		score: number;
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

	interface ServerState {
		stage: string;
		progress: Progress;
		stats: Stats | null;
		error: string | null;
	}

	// State
	let stage = $state<string>('idle');
	let directoryPath = $state('');
	let searchQuery = $state('');
	let searchResults = $state<SearchResult[]>([]);
	let progress = $state<Progress>({ phase: '', current: 0, total: 0 });
	let stats = $state<Stats | null>(null);
	let error = $state<string | null>(null);
	let isSearching = $state(false);

	// Derived
	let progressPercent = $derived(
		progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0
	);

	let isProcessing = $derived(
		stage === 'freezing' || stage === 'chunking' || stage === 'embedding'
	);

	let canSearch = $derived(stage === 'ready' && !isSearching);

	// SSE connection
	onMount(() => {
		const eventSource = new EventSource('/api/events');

		eventSource.onmessage = (event) => {
			const data: ServerState = JSON.parse(event.data);
			stage = data.stage;
			progress = data.progress;
			if (data.stats) {
				stats = data.stats;
			}
			error = data.error;
		};

		eventSource.onerror = () => {
			console.error('SSE connection error');
		};

		return () => {
			eventSource.close();
		};
	});

	// Handlers
	async function handleProcess() {
		if (!directoryPath.trim()) return;

		error = null;
		searchResults = [];

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

	async function handleSearch() {
		if (!searchQuery.trim() || !canSearch) return;

		isSearching = true;
		error = null;

		try {
			const response = await fetch('/api/search', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ query: searchQuery.trim(), k: 10 })
			});

			if (!response.ok) {
				const data = await response.json();
				error = data.detail || 'Search failed';
				return;
			}

			const data = await response.json();
			searchResults = data.results;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Network error';
		} finally {
			isSearching = false;
		}
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			if (stage === 'idle' || stage === 'ready' || stage === 'error') {
				if (searchQuery.trim() && canSearch) {
					handleSearch();
				} else if (directoryPath.trim() && !isProcessing) {
					handleProcess();
				}
			}
		}
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}

	function getPhaseLabel(phase: string): string {
		switch (phase) {
			case 'freezing': return 'Reading files';
			case 'chunking': return 'Chunking text';
			case 'embedding': return 'Generating embeddings';
			default: return phase;
		}
	}
</script>

<main class="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex flex-col items-center px-4 py-12">
	<div class="w-full max-w-4xl flex flex-col items-center gap-8">
		<!-- Logo/Title -->
		<h1 class="text-5xl font-bold text-white tracking-tight">
			Doctown
		</h1>
		<p class="text-white/60 text-lg -mt-4">
			Ask questions about your data
		</p>

		<!-- Error Display -->
		{#if error}
			<div class="w-full bg-red-500/20 border border-red-500/50 rounded-xl p-4 text-red-200">
				{error}
			</div>
		{/if}

		<!-- Path Input (show when idle or ready for new processing) -->
		{#if stage === 'idle' || stage === 'ready' || stage === 'error'}
			<div class="w-full">
				<label class="text-white/60 text-sm mb-2 block">Directory to analyze</label>
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
						Freeze
					</button>
				</div>
			</div>
		{/if}

		<!-- Progress Bar (show during processing) -->
		{#if isProcessing}
			<div class="w-full bg-white/5 rounded-xl p-6">
				<div class="flex items-center justify-between mb-3">
					<span class="text-white/80 font-medium">
						{getPhaseLabel(stage)}...
					</span>
					<span class="text-white/60 text-sm">
						{progress.current} / {progress.total}
					</span>
				</div>
				<div class="w-full bg-white/10 rounded-full h-3 overflow-hidden">
					<div
						class="h-full bg-blue-500 transition-all duration-300 ease-out"
						style="width: {progressPercent}%"
					></div>
				</div>
				<p class="text-white/40 text-sm mt-3">
					{#if stage === 'embedding'}
						This may take a moment...
					{:else}
						Processing your files...
					{/if}
				</p>
			</div>
		{/if}

		<!-- Search Bar (show when ready) -->
		{#if stage === 'ready'}
			<div class="w-full">
				<div class="relative">
					<input
						type="text"
						bind:value={searchQuery}
						onkeydown={handleKeydown}
						placeholder="Ask a question about your data..."
						disabled={isSearching}
						autofocus
						class="w-full px-6 py-5 text-xl bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all pr-16"
					/>
					<button
						onclick={handleSearch}
						disabled={!searchQuery.trim() || isSearching}
						class="absolute right-3 top-1/2 -translate-y-1/2 p-3 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-600/50 rounded-xl text-white transition-colors"
					>
						{#if isSearching}
							<svg class="h-6 w-6 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
						{:else}
							<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
							</svg>
						{/if}
					</button>
				</div>
			</div>
		{/if}

		<!-- Search Results -->
		{#if searchResults.length > 0}
			<div class="w-full space-y-4">
				<h2 class="text-white/60 text-sm font-medium">
					{searchResults.length} results
				</h2>
				{#each searchResults as result, i}
					<div class="bg-white/5 border border-white/10 rounded-xl p-5 hover:bg-white/[0.07] transition-colors">
						<div class="flex items-center justify-between mb-3">
							<div class="flex items-center gap-2">
								<span class="text-white/40 text-sm font-mono">
									[{i + 1}]
								</span>
								<span class="text-blue-400 font-medium">
									{result.file_path}
								</span>
							</div>
							<span class="text-white/40 text-sm">
								{(result.score * 100).toFixed(1)}% match
							</span>
						</div>
						<pre class="text-white/80 text-sm font-mono whitespace-pre-wrap overflow-x-auto bg-black/20 rounded-lg p-4">{result.text}</pre>
					</div>
				{/each}
			</div>
		{/if}

		<!-- Stats Panel (show when ready) -->
		{#if stats && (stage === 'ready' || isProcessing)}
			<div class="w-full flex items-center justify-center gap-8 text-white/50 text-sm">
				<div class="flex items-center gap-2">
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
					</svg>
					<span>{stats.total_files} files</span>
				</div>
				<div class="flex items-center gap-2">
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7" />
					</svg>
					<span>{stats.total_chunks} chunks</span>
				</div>
				<div class="flex items-center gap-2">
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
					</svg>
					<span>{stats.total_vectors} vectors</span>
				</div>
				{#if stats.total_size_bytes > 0}
					<div class="flex items-center gap-2">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
						</svg>
						<span>{formatSize(stats.total_size_bytes)}</span>
					</div>
				{/if}
			</div>
		{/if}

		<!-- Empty state hint -->
		{#if stage === 'idle'}
			<div class="text-center text-white/40 mt-8">
				<p>Enter a directory path above to create a searchable semantic snapshot.</p>
				<p class="mt-2 text-sm">Example: <code class="bg-white/10 px-2 py-1 rounded">~/Documents/my-project</code></p>
			</div>
		{/if}
	</div>
</main>
