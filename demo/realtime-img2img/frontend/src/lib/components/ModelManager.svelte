<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import Button from './Button.svelte';

  const dispatch = createEventDispatcher();

  let baseModels: string[] = [];
  let availableLoras: string[] = [];
  
  let selectedBaseModel: string = 'stabilityai/sd-turbo';
  let activeLoras: { name: string; weight: number }[] = [];
  
  let downloadUrl: string = '';
  let downloadName: string = '';
  let downloadType: 'model' | 'lora' = 'model';
  
  let isDownloading = false;
  let isLoadingModel = false;
  let statusMessage = '';

  onMount(async () => {
    await fetchModels();
  });

  async function fetchModels() {
    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      baseModels = data.base_models || [];
      availableLoras = data.loras || [];
      if (!baseModels.includes(selectedBaseModel) && baseModels.length > 0) {
        selectedBaseModel = baseModels[0];
      }
    } catch (e) {
      console.error(e);
      statusMessage = "Failed to fetch models";
    }
  }

  function addLora() {
    if (availableLoras.length > 0) {
      activeLoras = [...activeLoras, { name: availableLoras[0], weight: 1.0 }];
    }
  }

  function removeLora(index: number) {
    activeLoras = activeLoras.filter((_, i) => i !== index);
  }

  async function downloadModel() {
    if (!downloadUrl) return;
    isDownloading = true;
    statusMessage = "Downloading...";
    try {
      const res = await fetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: downloadUrl,
          model_name: downloadName,
          is_lora: downloadType === 'lora'
        })
      });
      const data = await res.json();
      if (res.ok) {
        statusMessage = "Download complete!";
        downloadUrl = '';
        downloadName = '';
        await fetchModels();
      } else {
        statusMessage = "Error: " + data.detail;
      }
    } catch (e) {
      console.error(e);
      statusMessage = "Download failed";
    } finally {
      isDownloading = false;
    }
  }

  async function applyModel() {
    isLoadingModel = true;
    statusMessage = "Loading model... This clears VRAM and may take a moment.";
    
    // Stop stream if it's running before loading to avoid errors
    dispatch('stopStream');

    const lora_dict: Record<string, number> = {};
    activeLoras.forEach(l => {
      lora_dict[l.name] = l.weight;
    });

    try {
      const res = await fetch('/api/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_model: selectedBaseModel,
          lora_dict: Object.keys(lora_dict).length > 0 ? lora_dict : null
        })
      });
      const data = await res.json();
      if (res.ok) {
        statusMessage = "Model loaded successfully! You can start the stream now.";
        dispatch('modelLoaded');
      } else {
        statusMessage = "Error: " + data.detail;
      }
    } catch (e) {
      console.error(e);
      statusMessage = "Failed to load model";
    } finally {
      isLoadingModel = false;
    }
  }
</script>

<div class="flex flex-col gap-4 rounded-lg bg-gray-100 p-4 dark:bg-gray-800">
  <h2 class="text-xl font-bold">Model Manager</h2>
  
  <!-- Base Model Selection -->
  <div class="flex flex-col gap-2">
    <label for="baseModel" class="font-semibold">Base Model</label>
    <select id="baseModel" bind:value={selectedBaseModel} class="rounded border p-2 dark:bg-gray-700">
      {#each baseModels as model}
        <option value={model}>{model}</option>
      {/each}
    </select>
  </div>

  <!-- LoRA Section -->
  <div class="flex flex-col gap-2">
    <div class="flex items-center justify-between">
      <span class="font-semibold">LoRAs</span>
      <Button on:click={addLora} classList="px-2 py-1 text-sm bg-blue-500 text-white rounded">+ Add LoRA</Button>
    </div>
    
    {#each activeLoras as lora, i}
      <div class="flex items-center gap-2 rounded border p-2 dark:bg-gray-700">
        <select bind:value={lora.name} class="w-full rounded p-1 dark:bg-gray-600">
          {#each availableLoras as al}
            <option value={al}>{al}</option>
          {/each}
        </select>
        
        <input type="number" step="0.1" bind:value={lora.weight} class="w-16 rounded p-1 dark:bg-gray-600" title="Weight" />
        
        <button on:click={() => removeLora(i)} class="text-red-500 hover:text-red-700" title="Remove">✕</button>
      </div>
    {/each}
    {#if activeLoras.length === 0}
      <p class="text-sm text-gray-500 italic">No LoRAs added.</p>
    {/if}
  </div>

  <Button on:click={applyModel} disabled={isLoadingModel} classList="bg-green-600 hover:bg-green-700 text-white font-bold py-2">
    {isLoadingModel ? 'Applying...' : 'Apply Models'}
  </Button>
  
  {#if statusMessage}
    <p class="text-sm {statusMessage.includes('Error') || statusMessage.includes('failed') ? 'text-red-500' : 'text-blue-500'}">{statusMessage}</p>
  {/if}

  <hr class="my-2 border-gray-300 dark:border-gray-600" />

  <!-- Download Model Section -->
  <div>
    <h3 class="font-semibold mb-2">Download Model / LoRA</h3>
    <div class="flex flex-col gap-2">
      <input type="text" bind:value={downloadUrl} placeholder="Direct URL (e.g. Civitai download link)" class="rounded border p-2 w-full dark:bg-gray-700" />
      <div class="flex gap-2">
        <input type="text" bind:value={downloadName} placeholder="Filename (optional)" class="rounded border p-2 w-2/3 dark:bg-gray-700" />
        <select bind:value={downloadType} class="rounded border p-2 w-1/3 dark:bg-gray-700">
          <option value="model">Checkpoint</option>
          <option value="lora">LoRA</option>
        </select>
      </div>
      <Button on:click={downloadModel} disabled={isDownloading || !downloadUrl} classList="bg-blue-600 hover:bg-blue-700 text-white py-1">
        {isDownloading ? 'Downloading...' : 'Download'}
      </Button>
    </div>
  </div>
</div>
