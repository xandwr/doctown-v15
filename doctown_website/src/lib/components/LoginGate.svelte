<script lang="ts">
	import { SignedIn, SignedOut, SignInButton } from 'svelte-clerk';

	interface Props {
		/** Message to show when user is not logged in */
		message?: string;
		/** Whether to show the sign-in button */
		showSignIn?: boolean;
		/** Additional CSS classes for the gate container */
		class?: string;
		/** The content to show when logged in */
		children: import('svelte').Snippet;
		/** Optional content to show when logged out (overrides default) */
		fallback?: import('svelte').Snippet;
	}

	let {
		message = 'Sign in to access this feature',
		showSignIn = true,
		class: className = '',
		children,
		fallback
	}: Props = $props();
</script>

<SignedIn>
	{@render children()}
</SignedIn>

<SignedOut>
	{#if fallback}
		{@render fallback()}
	{:else}
		<div class="flex flex-col items-center justify-center gap-3 py-8 px-4 {className}">
			<div class="flex items-center gap-2 text-white/50">
				<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
				</svg>
				<span class="text-sm">{message}</span>
			</div>
			{#if showSignIn}
				<SignInButton mode="modal">
					<button class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
						Sign In
					</button>
				</SignInButton>
			{/if}
		</div>
	{/if}
</SignedOut>
