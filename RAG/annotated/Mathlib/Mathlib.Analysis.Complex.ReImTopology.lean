/-- `Complex.re` turns `ℂ` into a trivial topological fiber bundle over `ℝ`. -/
theorem isHomeomorphicTrivialFiberBundle_re : IsHomeomorphicTrivialFiberBundle ℝ re :=
  ⟨equivRealProdCLM.toHomeomorph, fun _ => rfl⟩


/-- `Complex.im` turns `ℂ` into a trivial topological fiber bundle over `ℝ`. -/
theorem isHomeomorphicTrivialFiberBundle_im : IsHomeomorphicTrivialFiberBundle ℝ im :=
  ⟨equivRealProdCLM.toHomeomorph.trans (Homeomorph.prodComm ℝ ℝ), fun _ => rfl⟩


theorem isOpenMap_re : IsOpenMap re :=
  isHomeomorphicTrivialFiberBundle_re.isOpenMap_proj


theorem isOpenMap_im : IsOpenMap im :=
  isHomeomorphicTrivialFiberBundle_im.isOpenMap_proj


theorem isQuotientMap_re : IsQuotientMap re :=
  isHomeomorphicTrivialFiberBundle_re.isQuotientMap_proj


@[deprecated (since := "2024-10-22")]
alias quotientMap_re := isQuotientMap_re


theorem isQuotientMap_im : IsQuotientMap im :=
  isHomeomorphicTrivialFiberBundle_im.isQuotientMap_proj


@[deprecated (since := "2024-10-22")]
alias quotientMap_im := isQuotientMap_im


theorem interior_preimage_re (s : Set ℝ) : interior (re ⁻¹' s) = re ⁻¹' interior s :=
  (isOpenMap_re.preimage_interior_eq_interior_preimage continuous_re _).symm


theorem interior_preimage_im (s : Set ℝ) : interior (im ⁻¹' s) = im ⁻¹' interior s :=
  (isOpenMap_im.preimage_interior_eq_interior_preimage continuous_im _).symm


theorem closure_preimage_re (s : Set ℝ) : closure (re ⁻¹' s) = re ⁻¹' closure s :=
  (isOpenMap_re.preimage_closure_eq_closure_preimage continuous_re _).symm


theorem closure_preimage_im (s : Set ℝ) : closure (im ⁻¹' s) = im ⁻¹' closure s :=
  (isOpenMap_im.preimage_closure_eq_closure_preimage continuous_im _).symm


theorem frontier_preimage_re (s : Set ℝ) : frontier (re ⁻¹' s) = re ⁻¹' frontier s :=
  (isOpenMap_re.preimage_frontier_eq_frontier_preimage continuous_re _).symm


theorem frontier_preimage_im (s : Set ℝ) : frontier (im ⁻¹' s) = im ⁻¹' frontier s :=
  (isOpenMap_im.preimage_frontier_eq_frontier_preimage continuous_im _).symm


@[simp]
theorem interior_setOf_re_le (a : ℝ) : interior { z : ℂ | z.re ≤ a } = { z | z.re < a } := by
  /-
    a : Real
    ⊢ Eq (interior (setOf fun z => LE.le z.re a)) (setOf fun z => LT.lt z.re a)
  -/
  simpa only [interior_Iic] using interior_preimage_re (Iic a)
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_setOf_im_le (a : ℝ) : interior { z : ℂ | z.im ≤ a } = { z | z.im < a } := by
  /-
    a : Real
    ⊢ Eq (interior (setOf fun z => LE.le z.im a)) (setOf fun z => LT.lt z.im a)
  -/
  simpa only [interior_Iic] using interior_preimage_im (Iic a)
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_setOf_le_re (a : ℝ) : interior { z : ℂ | a ≤ z.re } = { z | a < z.re } := by
  /-
    a : Real
    ⊢ Eq (interior (setOf fun z => LE.le a z.re)) (setOf fun z => LT.lt a z.re)
  -/
  simpa only [interior_Ici] using interior_preimage_re (Ici a)
  /-
    🎉 no goals
  -/


@[simp]
theorem interior_setOf_le_im (a : ℝ) : interior { z : ℂ | a ≤ z.im } = { z | a < z.im } := by
  /-
    a : Real
    ⊢ Eq (interior (setOf fun z => LE.le a z.im)) (setOf fun z => LT.lt a z.im)
  -/
  simpa only [interior_Ici] using interior_preimage_im (Ici a)
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_setOf_re_lt (a : ℝ) : closure { z : ℂ | z.re < a } = { z | z.re ≤ a } := by
  /-
    a : Real
    ⊢ Eq (closure (setOf fun z => LT.lt z.re a)) (setOf fun z => LE.le z.re a)
  -/
  simpa only [closure_Iio] using closure_preimage_re (Iio a)
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_setOf_im_lt (a : ℝ) : closure { z : ℂ | z.im < a } = { z | z.im ≤ a } := by
  /-
    a : Real
    ⊢ Eq (closure (setOf fun z => LT.lt z.im a)) (setOf fun z => LE.le z.im a)
  -/
  simpa only [closure_Iio] using closure_preimage_im (Iio a)
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_setOf_lt_re (a : ℝ) : closure { z : ℂ | a < z.re } = { z | a ≤ z.re } := by
  /-
    a : Real
    ⊢ Eq (closure (setOf fun z => LT.lt a z.re)) (setOf fun z => LE.le a z.re)
  -/
  simpa only [closure_Ioi] using closure_preimage_re (Ioi a)
  /-
    🎉 no goals
  -/


@[simp]
theorem closure_setOf_lt_im (a : ℝ) : closure { z : ℂ | a < z.im } = { z | a ≤ z.im } := by
  /-
    a : Real
    ⊢ Eq (closure (setOf fun z => LT.lt a z.im)) (setOf fun z => LE.le a z.im)
  -/
  simpa only [closure_Ioi] using closure_preimage_im (Ioi a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_re_le (a : ℝ) : frontier { z : ℂ | z.re ≤ a } = { z | z.re = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LE.le z.re a)) (setOf fun z => Eq z.re a)
  -/
  simpa only [frontier_Iic] using frontier_preimage_re (Iic a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_im_le (a : ℝ) : frontier { z : ℂ | z.im ≤ a } = { z | z.im = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LE.le z.im a)) (setOf fun z => Eq z.im a)
  -/
  simpa only [frontier_Iic] using frontier_preimage_im (Iic a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_le_re (a : ℝ) : frontier { z : ℂ | a ≤ z.re } = { z | z.re = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LE.le a z.re)) (setOf fun z => Eq z.re a)
  -/
  simpa only [frontier_Ici] using frontier_preimage_re (Ici a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_le_im (a : ℝ) : frontier { z : ℂ | a ≤ z.im } = { z | z.im = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LE.le a z.im)) (setOf fun z => Eq z.im a)
  -/
  simpa only [frontier_Ici] using frontier_preimage_im (Ici a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_re_lt (a : ℝ) : frontier { z : ℂ | z.re < a } = { z | z.re = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LT.lt z.re a)) (setOf fun z => Eq z.re a)
  -/
  simpa only [frontier_Iio] using frontier_preimage_re (Iio a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_im_lt (a : ℝ) : frontier { z : ℂ | z.im < a } = { z | z.im = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LT.lt z.im a)) (setOf fun z => Eq z.im a)
  -/
  simpa only [frontier_Iio] using frontier_preimage_im (Iio a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_lt_re (a : ℝ) : frontier { z : ℂ | a < z.re } = { z | z.re = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LT.lt a z.re)) (setOf fun z => Eq z.re a)
  -/
  simpa only [frontier_Ioi] using frontier_preimage_re (Ioi a)
  /-
    🎉 no goals
  -/


@[simp]
theorem frontier_setOf_lt_im (a : ℝ) : frontier { z : ℂ | a < z.im } = { z | z.im = a } := by
  /-
    a : Real
    ⊢ Eq (frontier (setOf fun z => LT.lt a z.im)) (setOf fun z => Eq z.im a)
  -/
  simpa only [frontier_Ioi] using frontier_preimage_im (Ioi a)
  /-
    🎉 no goals
  -/


theorem closure_reProdIm (s t : Set ℝ) : closure (s ×ℂ t) = closure s ×ℂ closure t := by
  simpa only [← preimage_eq_preimage equivRealProdCLM.symm.toHomeomorph.surjective,
    equivRealProdCLM.symm.toHomeomorph.preimage_closure] using @closure_prod_eq _ _ _ _ s t


theorem interior_reProdIm (s t : Set ℝ) : interior (s ×ℂ t) = interior s ×ℂ interior t := by
  /-
    s t : Set Real
    ⊢ Eq (interior (Complex.reProdIm s t)) (Complex.reProdIm (interior s) (interio …
  -/
  rw [reProdIm, reProdIm, interior_inter, interior_preimage_re, interior_preimage_im]
  /-
    🎉 no goals
  -/


theorem frontier_reProdIm (s t : Set ℝ) :
    frontier (s ×ℂ t) = closure s ×ℂ frontier t ∪ frontier s ×ℂ closure t := by
  simpa only [← preimage_eq_preimage equivRealProdCLM.symm.toHomeomorph.surjective,
    equivRealProdCLM.symm.toHomeomorph.preimage_frontier] using frontier_prod_eq s t


theorem frontier_setOf_le_re_and_le_im (a b : ℝ) :
    frontier { z | a ≤ re z ∧ b ≤ im z } = { z | a ≤ re z ∧ im z = b ∨ re z = a ∧ b ≤ im z } := by
  /-
    a b : Real
    ⊢ Eq (frontier (setOf fun z => And (LE.le a z.re) (LE.le b z.im))) (setOf fun  …
  -/
  simpa only [closure_Ici, frontier_Ici] using frontier_reProdIm (Ici a) (Ici b)
  /-
    🎉 no goals
  -/


theorem frontier_setOf_le_re_and_im_le (a b : ℝ) :
    frontier { z | a ≤ re z ∧ im z ≤ b } = { z | a ≤ re z ∧ im z = b ∨ re z = a ∧ im z ≤ b } := by
  simpa only [closure_Ici, closure_Iic, frontier_Ici, frontier_Iic] using
    frontier_reProdIm (Ici a) (Iic b)


theorem IsOpen.reProdIm (hs : IsOpen s) (ht : IsOpen t) : IsOpen (s ×ℂ t) :=
  (hs.preimage continuous_re).inter (ht.preimage continuous_im)


theorem IsClosed.reProdIm (hs : IsClosed s) (ht : IsClosed t) : IsClosed (s ×ℂ t) :=
  (hs.preimage continuous_re).inter (ht.preimage continuous_im)


theorem Bornology.IsBounded.reProdIm (hs : IsBounded s) (ht : IsBounded t) : IsBounded (s ×ℂ t) :=
  antilipschitz_equivRealProd.isBounded_preimage (hs.prod ht)


protected lemma TendstoUniformlyOn.re {f : ι → α → ℂ} {p : Filter ι} {g : α → ℂ} {K : Set α}
    (hf : TendstoUniformlyOn f g p K) :
    TendstoUniformlyOn (fun n x => (f n x).re) (fun y => (g y).re) p K := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    p : Filter ι
    g : α → Complex
    K : Set α
    hf : TendstoUniformlyOn f g p K
    ⊢ TendstoUniformlyOn (fun n x => (f n x).re) (fun y => (g y).re) p K
  -/
  apply UniformContinuous.comp_tendstoUniformlyOn uniformlyContinuous_re hf
  /-
    🎉 no goals
  -/


protected lemma TendstoUniformly.re {f : ι → α → ℂ} {p : Filter ι} {g : α → ℂ}
    (hf : TendstoUniformly f g p) :
    TendstoUniformly (fun n x => (f n x).re) (fun y => (g y).re) p := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    p : Filter ι
    g : α → Complex
    hf : TendstoUniformly f g p
    ⊢ TendstoUniformly (fun n x => (f n x).re) (fun y => (g y).re) p
  -/
  apply UniformContinuous.comp_tendstoUniformly uniformlyContinuous_re hf
  /-
    🎉 no goals
  -/


protected lemma TendstoUniformlyOn.im {f : ι → α → ℂ} {p : Filter ι} {g : α → ℂ} {K : Set α}
    (hf : TendstoUniformlyOn f g p K) :
    TendstoUniformlyOn (fun n x => (f n x).im) (fun y => (g y).im) p K := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    p : Filter ι
    g : α → Complex
    K : Set α
    hf : TendstoUniformlyOn f g p K
    ⊢ TendstoUniformlyOn (fun n x => (f n x).im) (fun y => (g y).im) p K
  -/
  apply UniformContinuous.comp_tendstoUniformlyOn uniformlyContinuous_im hf
  /-
    🎉 no goals
  -/


protected lemma TendstoUniformly.im {f : ι → α → ℂ} {p : Filter ι} {g : α → ℂ}
    (hf : TendstoUniformly f g p) :
    TendstoUniformly (fun n x => (f n x).im) (fun y => (g y).im) p := by
  /-
    α : Type u_1
    ι : Type u_2
    f : ι → α → Complex
    p : Filter ι
    g : α → Complex
    hf : TendstoUniformly f g p
    ⊢ TendstoUniformly (fun n x => (f n x).im) (fun y => (g y).im) p
  -/
  apply UniformContinuous.comp_tendstoUniformly uniformlyContinuous_im hf
  /-
    🎉 no goals
  -/


