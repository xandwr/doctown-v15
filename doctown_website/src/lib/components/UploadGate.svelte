<script lang="ts">
	import { SignedIn, SignedOut, SignInButton } from 'svelte-clerk';

	interface Props {
		/** Drag over state from parent */
		isDragOver?: boolean;
	}

	let { isDragOver = false }: Props = $props();
</script>

<!-- Visible Upload Zone (for logged-in users OR as preview for anon) -->
<SignedIn>
	<slot />
</SignedIn>

<SignedOut>
	<!-- Disabled preview upload zone for anonymous users -->
	<div
		class="relative border-2 border-dashed rounded-xl sm:rounded-2xl p-6 sm:p-8 text-center transition-all
			border-white/10 bg-white/2 cursor-not-allowed"
	>
		<div class="flex flex-col items-center gap-2 sm:gap-3 opacity-50">
			<svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 sm:h-10 sm:w-10 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
			</svg>
			<div>
				<p class="text-white/70 text-base sm:text-lg">Drop files here or tap to browse</p>
				<p class="text-white/40 text-xs sm:text-sm mt-1">Upload documents, code, or any text files</p>
			</div>
		</div>

		<!-- Login overlay -->
		<div class="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm rounded-xl sm:rounded-2xl">
			<div class="flex items-center gap-2 text-white/70 mb-3">
				<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
				</svg>
				<span class="text-sm font-medium">Sign in to upload files</span>
			</div>
			<SignInButton mode="modal">
				<button class="px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-xl transition-colors shadow-lg shadow-blue-600/20">
					Sign In to Get Started
				</button>
			</SignInButton>
			<p class="text-white/40 text-xs mt-3">Free tier includes limited processing</p>
		</div>
	</div>
</SignedOut>
