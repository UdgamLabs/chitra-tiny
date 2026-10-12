/* Local presentation only: switches between saved images. No inference, analytics, fetches or storage. */
(() => {
  'use strict';
  const notes = {
    1: 'Guidance 1 means no extra push towards the prompt. The picture is loose and only roughly a snowy cottage.',
    3: 'A little guidance: a clearer cottage and forest, still with some variety.',
    5: 'The scene is now plainly the prompt, with crisper shapes.',
    7: 'The default. It scored best on GenEval: the subject is clear and the scene is clean.',
    10: 'Very high guidance follows the words literally and pushes colour and contrast harder.'
  };
  function setup(groupId, imgId, file, alt, noteId) {
    const buttons = [...document.querySelectorAll(`#${groupId} button`)];
    const img = document.getElementById(imgId);
    const note = noteId && document.getElementById(noteId);
    function show(i) {
      const v = buttons[i].dataset.value;
      img.src = file(v); img.alt = alt(v);
      if (note) note.textContent = notes[v];
      buttons.forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
    }
    buttons.forEach((b, i) => {
      b.addEventListener('click', () => show(i));
      b.addEventListener('keydown', e => {
        let n;
        if (e.key === 'ArrowRight') n = (i + 1) % buttons.length;
        else if (e.key === 'ArrowLeft') n = (i + buttons.length - 1) % buttons.length;
        else return;
        e.preventDefault(); buttons[n].focus(); show(n);
      });
    });
  }
  setup('guidance-dial', 'guidance-img', v => `assets/img/guidance_${String(v).padStart(2, '0')}.jpg`,
        v => `“a cottage in a snowy forest”, seed 0, guidance ${v}`, 'guidance-note');
  setup('seed-dial', 'seed-img', v => `assets/img/seed_${v}.jpg`,
        v => `“an old stone church in a green valley”, seed ${v}, guidance 7`);
})();
