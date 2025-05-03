/-- `n`-times continuously differentiable diffeomorphism between `M` and `M'` with respect to `I`
and `I'`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): was @[nolint has_nonempty_instance]
structure Diffeomorph extends M ≃ M' where
  protected contMDiff_toFun : ContMDiff I I' n toEquiv
  protected contMDiff_invFun : ContMDiff I' I n toEquiv.symm


@[inherit_doc]
scoped[Manifold] notation M " ≃ₘ^" n:1000 "⟮" I ", " J "⟯ " N => Diffeomorph I J M N n


/-- Infinitely differentiable diffeomorphism between `M` and `M'` with respect to `I` and `I'`. -/
scoped[Manifold] notation M " ≃ₘ⟮" I ", " J "⟯ " N => Diffeomorph I J M N ⊤


/-- `n`-times continuously differentiable diffeomorphism between `E` and `E'`. -/
scoped[Manifold] notation E " ≃ₘ^" n:1000 "[" 𝕜 "] " E' => Diffeomorph 𝓘(𝕜, E) 𝓘(𝕜, E') E E' n


/-- Infinitely differentiable diffeomorphism between `E` and `E'`. -/
scoped[Manifold]
  notation3 E " ≃ₘ[" 𝕜 "] " E' =>
    Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' ⊤


theorem toEquiv_injective : Injective (Diffeomorph.toEquiv : (M ≃ₘ^n⟮I, I'⟯ M') → M ≃ M')
  | ⟨_, _, _⟩, ⟨_, _, _⟩, rfl => rfl


instance : EquivLike (M ≃ₘ^n⟮I, I'⟯ M') M M' where
  coe Φ := Φ.toEquiv
  inv Φ := Φ.toEquiv.symm
  left_inv Φ := Φ.left_inv
  right_inv Φ := Φ.right_inv
  coe_injective' _ _ h _ := toEquiv_injective <| DFunLike.ext' h


/-- Interpret a diffeomorphism as a `ContMDiffMap`. -/
@[coe]
def toContMDiffMap (Φ : M ≃ₘ^n⟮I, I'⟯ M') : C^n⟮I, M; I', M'⟯ :=
  ⟨Φ, Φ.contMDiff_toFun⟩


instance : Coe (M ≃ₘ^n⟮I, I'⟯ M') C^n⟮I, M; I', M'⟯ :=
  ⟨toContMDiffMap⟩


@[continuity]
protected theorem continuous (h : M ≃ₘ^n⟮I, I'⟯ M') : Continuous h :=
  h.contMDiff_toFun.continuous


protected theorem contMDiff (h : M ≃ₘ^n⟮I, I'⟯ M') : ContMDiff I I' n h :=
  h.contMDiff_toFun


protected theorem contMDiffAt (h : M ≃ₘ^n⟮I, I'⟯ M') {x} : ContMDiffAt I I' n h x :=
  h.contMDiff.contMDiffAt


protected theorem contMDiffWithinAt (h : M ≃ₘ^n⟮I, I'⟯ M') {s x} : ContMDiffWithinAt I I' n h s x :=
  h.contMDiffAt.contMDiffWithinAt

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: should use `E ≃ₘ^n[𝕜] F` notation

protected theorem contDiff (h : E ≃ₘ^n⟮𝓘(𝕜, E), 𝓘(𝕜, E')⟯ E') : ContDiff 𝕜 n h :=
  h.contMDiff.contDiff


@[deprecated (since := "2024-11-21")] alias smooth := Diffeomorph.contDiff


protected theorem mdifferentiable (h : M ≃ₘ^n⟮I, I'⟯ M') (hn : 1 ≤ n) : MDifferentiable I I' h :=
  h.contMDiff.mdifferentiable hn


protected theorem mdifferentiableOn (h : M ≃ₘ^n⟮I, I'⟯ M') (s : Set M) (hn : 1 ≤ n) :
    MDifferentiableOn I I' h s :=
  (h.mdifferentiable hn).mdifferentiableOn


@[simp]
theorem coe_toEquiv (h : M ≃ₘ^n⟮I, I'⟯ M') : ⇑h.toEquiv = h :=
  rfl


@[simp, norm_cast]
theorem coe_coe (h : M ≃ₘ^n⟮I, I'⟯ M') : ⇑(h : C^n⟮I, M; I', M'⟯) = h :=
  rfl


@[simp]
theorem toEquiv_inj {h h' : M ≃ₘ^n⟮I, I'⟯ M'} : h.toEquiv = h'.toEquiv ↔ h = h' :=
  toEquiv_injective.eq_iff


/-- Coercion to function `fun h : M ≃ₘ^n⟮I, I'⟯ M' ↦ (h : M → M')` is injective. -/
theorem coeFn_injective : Injective ((↑) : (M ≃ₘ^n⟮I, I'⟯ M') → (M → M')) :=
  DFunLike.coe_injective


@[ext]
theorem ext {h h' : M ≃ₘ^n⟮I, I'⟯ M'} (Heq : ∀ x, h x = h' x) : h = h' :=
  coeFn_injective <| funext Heq


instance : ContinuousMapClass (M ≃ₘ⟮I, J⟯ N) M N where
  map_continuous f := f.continuous


/-- Identity map as a diffeomorphism. -/
protected def refl : M ≃ₘ^n⟮I, I⟯ M where
  contMDiff_toFun := contMDiff_id
  contMDiff_invFun := contMDiff_id
  toEquiv := Equiv.refl M


@[simp]
theorem refl_toEquiv : (Diffeomorph.refl I M n).toEquiv = Equiv.refl _ :=
  rfl


@[simp]
theorem coe_refl : ⇑(Diffeomorph.refl I M n) = id :=
  rfl


/-- Composition of two diffeomorphisms. -/
@[trans]
protected def trans (h₁ : M ≃ₘ^n⟮I, I'⟯ M') (h₂ : M' ≃ₘ^n⟮I', J⟯ N) : M ≃ₘ^n⟮I, J⟯ N where
  contMDiff_toFun := h₂.contMDiff.comp h₁.contMDiff
  contMDiff_invFun := h₁.contMDiff_invFun.comp h₂.contMDiff_invFun
  toEquiv := h₁.toEquiv.trans h₂.toEquiv


@[simp]
theorem trans_refl (h : M ≃ₘ^n⟮I, I'⟯ M') : h.trans (Diffeomorph.refl I' M' n) = h :=
  ext fun _ => rfl


@[simp]
theorem refl_trans (h : M ≃ₘ^n⟮I, I'⟯ M') : (Diffeomorph.refl I M n).trans h = h :=
  ext fun _ => rfl


@[simp]
theorem coe_trans (h₁ : M ≃ₘ^n⟮I, I'⟯ M') (h₂ : M' ≃ₘ^n⟮I', J⟯ N) : ⇑(h₁.trans h₂) = h₂ ∘ h₁ :=
  rfl


/-- Inverse of a diffeomorphism. -/
@[symm]
protected def symm (h : M ≃ₘ^n⟮I, J⟯ N) : N ≃ₘ^n⟮J, I⟯ M where
  contMDiff_toFun := h.contMDiff_invFun
  contMDiff_invFun := h.contMDiff_toFun
  toEquiv := h.toEquiv.symm


@[simp]
theorem apply_symm_apply (h : M ≃ₘ^n⟮I, J⟯ N) (x : N) : h (h.symm x) = x :=
  h.toEquiv.apply_symm_apply x


@[simp]
theorem symm_apply_apply (h : M ≃ₘ^n⟮I, J⟯ N) (x : M) : h.symm (h x) = x :=
  h.toEquiv.symm_apply_apply x


@[simp]
theorem symm_refl : (Diffeomorph.refl I M n).symm = Diffeomorph.refl I M n :=
  ext fun _ => rfl


@[simp]
theorem self_trans_symm (h : M ≃ₘ^n⟮I, J⟯ N) : h.trans h.symm = Diffeomorph.refl I M n :=
  ext h.symm_apply_apply


@[simp]
theorem symm_trans_self (h : M ≃ₘ^n⟮I, J⟯ N) : h.symm.trans h = Diffeomorph.refl J N n :=
  ext h.apply_symm_apply


@[simp]
theorem symm_trans' (h₁ : M ≃ₘ^n⟮I, I'⟯ M') (h₂ : M' ≃ₘ^n⟮I', J⟯ N) :
    (h₁.trans h₂).symm = h₂.symm.trans h₁.symm :=
  rfl


@[simp]
theorem symm_toEquiv (h : M ≃ₘ^n⟮I, J⟯ N) : h.symm.toEquiv = h.toEquiv.symm :=
  rfl


@[simp, mfld_simps]
theorem toEquiv_coe_symm (h : M ≃ₘ^n⟮I, J⟯ N) : ⇑h.toEquiv.symm = h.symm :=
  rfl


theorem image_eq_preimage (h : M ≃ₘ^n⟮I, J⟯ N) (s : Set M) : h '' s = h.symm ⁻¹' s :=
  h.toEquiv.image_eq_preimage s


theorem symm_image_eq_preimage (h : M ≃ₘ^n⟮I, J⟯ N) (s : Set N) : h.symm '' s = h ⁻¹' s :=
  h.symm.image_eq_preimage s


@[simp, mfld_simps]
nonrec theorem range_comp {α} (h : M ≃ₘ^n⟮I, J⟯ N) (f : α → M) :
    range (h ∘ f) = h.symm ⁻¹' range f := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_5
    inst✝⁵ : TopologicalSpace H
    G : Type u_7
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_9
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_11
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    α : Sort u_13
    h : Diffeomorph I J M N n
    f : α → M
    ⊢ Eq (Set.range (Function.comp (⇑h) f)) (Set.preimage (⇑h.symm) (Set.range f))
  -/
  rw [range_comp, image_eq_preimage]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_symm_image (h : M ≃ₘ^n⟮I, J⟯ N) (s : Set N) : h '' (h.symm '' s) = s :=
  h.toEquiv.image_symm_image s


@[simp]
theorem symm_image_image (h : M ≃ₘ^n⟮I, J⟯ N) (s : Set M) : h.symm '' (h '' s) = s :=
  h.toEquiv.symm_image_image s


/-- A diffeomorphism is a homeomorphism. -/
def toHomeomorph (h : M ≃ₘ^n⟮I, J⟯ N) : M ≃ₜ N :=
  ⟨h.toEquiv, h.continuous, h.symm.continuous⟩


@[simp]
theorem toHomeomorph_toEquiv (h : M ≃ₘ^n⟮I, J⟯ N) : h.toHomeomorph.toEquiv = h.toEquiv :=
  rfl


@[simp]
theorem symm_toHomeomorph (h : M ≃ₘ^n⟮I, J⟯ N) : h.symm.toHomeomorph = h.toHomeomorph.symm :=
  rfl


@[simp]
theorem coe_toHomeomorph (h : M ≃ₘ^n⟮I, J⟯ N) : ⇑h.toHomeomorph = h :=
  rfl


@[simp]
theorem coe_toHomeomorph_symm (h : M ≃ₘ^n⟮I, J⟯ N) : ⇑h.toHomeomorph.symm = h.symm :=
  rfl


@[simp]
theorem contMDiffWithinAt_comp_diffeomorph_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : N → M'} {s x}
    (hm : m ≤ n) :
    ContMDiffWithinAt I I' m (f ∘ h) s x ↔ ContMDiffWithinAt J I' m f (h.symm ⁻¹' s) (h x) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁴ : NormedAddCommGroup E
    inst✝¹³ : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹² : NormedAddCommGroup E'
    inst✝¹¹ : NormedSpace 𝕜 E'
    F : Type u_4
    inst✝¹⁰ : NormedAddCommGroup F
    inst✝⁹ : NormedSpace 𝕜 F
    H : Type u_5
    inst✝⁸ : TopologicalSpace H
    H' : Type u_6
    inst✝⁷ : TopologicalSpace H'
    G : Type u_7
    inst✝⁶ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    J : ModelWithCorners 𝕜 F G
    M : Type u_9
    inst✝⁵ : TopologicalSpace M
    inst✝⁴ : ChartedSpace H M
    M' : Type u_10
    inst✝³ : TopologicalSpace M'
    inst✝² : ChartedSpace H' M'
    N : Type u_11
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n m : ENat
    h : Diffeomorph I J M N n
    f : N → M'
    s : Set M
    x : M
    hm : LE.le m n
    ⊢ Iff (ContMDiffWithinAt I I' m (Function.comp f ⇑h) s x) (ContMDiffWithinAt J …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace 𝕜 F
      H : Type u_5
      inst✝⁸ : TopologicalSpace H
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      G : Type u_7
      inst✝⁶ : TopologicalSpace G
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      M : Type u_9
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      M' : Type u_10
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      N : Type u_11
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n m : ENat
      h : Diffeomorph I J M N n
      f : N → M'
      s : Set M
      x : M
      hm : LE.le m n
      ⊢ ContMDiffWithinAt I I' m (Function.comp f ⇑h) s x → ContMDiffWithinAt J I' m …
    -/
  · intro Hfh
    /-
      case mp
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace 𝕜 F
      H : Type u_5
      inst✝⁸ : TopologicalSpace H
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      G : Type u_7
      inst✝⁶ : TopologicalSpace G
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      M : Type u_9
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      M' : Type u_10
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      N : Type u_11
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n m : ENat
      h : Diffeomorph I J M N n
      f : N → M'
      s : Set M
      x : M
      hm : LE.le m n
      Hfh : ContMDiffWithinAt I I' m (Function.comp f ⇑h) s x
      ⊢ ContMDiffWithinAt J I' m f (Set.preimage (⇑h.symm) s) (h x)
    -/
    rw [← h.symm_apply_apply x] at Hfh
    simpa only [Function.comp_def, h.apply_symm_apply] using
      Hfh.comp (h x) (h.symm.contMDiffWithinAt.of_le hm) (mapsTo_preimage _ _)
    /-
      case mpr
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace 𝕜 F
      H : Type u_5
      inst✝⁸ : TopologicalSpace H
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      G : Type u_7
      inst✝⁶ : TopologicalSpace G
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      M : Type u_9
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      M' : Type u_10
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      N : Type u_11
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n m : ENat
      h : Diffeomorph I J M N n
      f : N → M'
      s : Set M
      x : M
      hm : LE.le m n
      ⊢ ContMDiffWithinAt J I' m f (Set.preimage (⇑h.symm) s) (h x) → ContMDiffWithi …
    -/
  · rw [← h.image_eq_preimage]
    /-
      case mpr
      𝕜 : Type u_1
      inst✝¹⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁴ : NormedAddCommGroup E
      inst✝¹³ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹² : NormedAddCommGroup E'
      inst✝¹¹ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹⁰ : NormedAddCommGroup F
      inst✝⁹ : NormedSpace 𝕜 F
      H : Type u_5
      inst✝⁸ : TopologicalSpace H
      H' : Type u_6
      inst✝⁷ : TopologicalSpace H'
      G : Type u_7
      inst✝⁶ : TopologicalSpace G
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      M : Type u_9
      inst✝⁵ : TopologicalSpace M
      inst✝⁴ : ChartedSpace H M
      M' : Type u_10
      inst✝³ : TopologicalSpace M'
      inst✝² : ChartedSpace H' M'
      N : Type u_11
      inst✝¹ : TopologicalSpace N
      inst✝ : ChartedSpace G N
      n m : ENat
      h : Diffeomorph I J M N n
      f : N → M'
      s : Set M
      x : M
      hm : LE.le m n
      ⊢ ContMDiffWithinAt J I' m f (Set.image (⇑h) s) (h x) → ContMDiffWithinAt I I' …
    -/
    exact fun hf => hf.comp x (h.contMDiffWithinAt.of_le hm) (mapsTo_image _ _)
    /-
      🎉 no goals
    -/


@[simp]
theorem contMDiffOn_comp_diffeomorph_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : N → M'} {s} (hm : m ≤ n) :
    ContMDiffOn I I' m (f ∘ h) s ↔ ContMDiffOn J I' m f (h.symm ⁻¹' s) :=
  h.toEquiv.forall_congr fun {_} => by
    simp only [hm, coe_toEquiv, h.symm_apply_apply, contMDiffWithinAt_comp_diffeomorph_iff,
      mem_preimage]


@[simp]
theorem contMDiffAt_comp_diffeomorph_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : N → M'} {x} (hm : m ≤ n) :
    ContMDiffAt I I' m (f ∘ h) x ↔ ContMDiffAt J I' m f (h x) :=
  h.contMDiffWithinAt_comp_diffeomorph_iff hm


@[simp]
theorem contMDiff_comp_diffeomorph_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : N → M'} (hm : m ≤ n) :
    ContMDiff I I' m (f ∘ h) ↔ ContMDiff J I' m f :=
  h.toEquiv.forall_congr fun _ ↦ h.contMDiffAt_comp_diffeomorph_iff hm


@[simp]
theorem contMDiffWithinAt_diffeomorph_comp_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : M' → M} (hm : m ≤ n)
    {s x} : ContMDiffWithinAt I' J m (h ∘ f) s x ↔ ContMDiffWithinAt I' I m f s x :=
  ⟨fun Hhf => by
    simpa only [Function.comp_def, h.symm_apply_apply] using
      (h.symm.contMDiffAt.of_le hm).comp_contMDiffWithinAt _ Hhf,
    fun Hf => (h.contMDiffAt.of_le hm).comp_contMDiffWithinAt _ Hf⟩


@[simp]
theorem contMDiffAt_diffeomorph_comp_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : M' → M} (hm : m ≤ n) {x} :
    ContMDiffAt I' J m (h ∘ f) x ↔ ContMDiffAt I' I m f x :=
  h.contMDiffWithinAt_diffeomorph_comp_iff hm


@[simp]
theorem contMDiffOn_diffeomorph_comp_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : M' → M} (hm : m ≤ n) {s} :
    ContMDiffOn I' J m (h ∘ f) s ↔ ContMDiffOn I' I m f s :=
  forall₂_congr fun _ _ => h.contMDiffWithinAt_diffeomorph_comp_iff hm


@[simp]
theorem contMDiff_diffeomorph_comp_iff {m} (h : M ≃ₘ^n⟮I, J⟯ N) {f : M' → M} (hm : m ≤ n) :
    ContMDiff I' J m (h ∘ f) ↔ ContMDiff I' I m f :=
  forall_congr' fun _ => h.contMDiffWithinAt_diffeomorph_comp_iff hm


theorem toPartialHomeomorph_mdifferentiable (h : M ≃ₘ^n⟮I, J⟯ N) (hn : 1 ≤ n) :
    h.toHomeomorph.toPartialHomeomorph.MDifferentiable I J :=
  ⟨h.mdifferentiableOn _ hn, h.symm.mdifferentiableOn _ hn⟩


/-- Product of two diffeomorphisms. -/
def prodCongr (h₁ : M ≃ₘ^n⟮I, I'⟯ M') (h₂ : N ≃ₘ^n⟮J, J'⟯ N') :
    (M × N) ≃ₘ^n⟮I.prod J, I'.prod J'⟯ M' × N' where
  contMDiff_toFun := (h₁.contMDiff.comp contMDiff_fst).prod_mk (h₂.contMDiff.comp contMDiff_snd)
  contMDiff_invFun :=
    (h₁.symm.contMDiff.comp contMDiff_fst).prod_mk (h₂.symm.contMDiff.comp contMDiff_snd)
  toEquiv := h₁.toEquiv.prodCongr h₂.toEquiv


@[simp]
theorem prodCongr_symm (h₁ : M ≃ₘ^n⟮I, I'⟯ M') (h₂ : N ≃ₘ^n⟮J, J'⟯ N') :
    (h₁.prodCongr h₂).symm = h₁.symm.prodCongr h₂.symm :=
  rfl


@[simp]
theorem coe_prodCongr (h₁ : M ≃ₘ^n⟮I, I'⟯ M') (h₂ : N ≃ₘ^n⟮J, J'⟯ N') :
    ⇑(h₁.prodCongr h₂) = Prod.map h₁ h₂ :=
  rfl


/-- `M × N` is diffeomorphic to `N × M`. -/
def prodComm : (M × N) ≃ₘ^n⟮I.prod J, J.prod I⟯ N × M where
  contMDiff_toFun := contMDiff_snd.prod_mk contMDiff_fst
  contMDiff_invFun := contMDiff_snd.prod_mk contMDiff_fst
  toEquiv := Equiv.prodComm M N


@[simp]
theorem prodComm_symm : (prodComm I J M N n).symm = prodComm J I N M n :=
  rfl


@[simp]
theorem coe_prodComm : ⇑(prodComm I J M N n) = Prod.swap :=
  rfl


/-- `(M × N) × N'` is diffeomorphic to `M × (N × N')`. -/
def prodAssoc : ((M × N) × N') ≃ₘ^n⟮(I.prod J).prod J', I.prod (J.prod J')⟯ M × N × N' where
  contMDiff_toFun :=
    (contMDiff_fst.comp contMDiff_fst).prod_mk
      ((contMDiff_snd.comp contMDiff_fst).prod_mk contMDiff_snd)
  contMDiff_invFun :=
    (contMDiff_fst.prod_mk (contMDiff_fst.comp contMDiff_snd)).prod_mk
      (contMDiff_snd.comp contMDiff_snd)
  toEquiv := Equiv.prodAssoc M N N'


theorem uniqueMDiffOn_image_aux (h : M ≃ₘ^n⟮I, J⟯ N) (hn : 1 ≤ n) {s : Set M}
    (hs : UniqueMDiffOn I s) : UniqueMDiffOn J (h '' s) := by
  /-
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_5
    inst✝⁵ : TopologicalSpace H
    G : Type u_7
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_9
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_11
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    h : Diffeomorph I J M N n
    hn : LE.le 1 n
    s : Set M
    hs : UniqueMDiffOn I s
    ⊢ UniqueMDiffOn J (Set.image (⇑h) s)
  -/
  convert hs.uniqueMDiffOn_preimage (h.toPartialHomeomorph_mdifferentiable hn)
  /-
    case h.e'_12
    𝕜 : Type u_1
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜 F
    H : Type u_5
    inst✝⁵ : TopologicalSpace H
    G : Type u_7
    inst✝⁴ : TopologicalSpace G
    I : ModelWithCorners 𝕜 E H
    J : ModelWithCorners 𝕜 F G
    M : Type u_9
    inst✝³ : TopologicalSpace M
    inst✝² : ChartedSpace H M
    N : Type u_11
    inst✝¹ : TopologicalSpace N
    inst✝ : ChartedSpace G N
    n : ENat
    h : Diffeomorph I J M N n
    hn : LE.le 1 n
    s : Set M
    hs : UniqueMDiffOn I s
    ⊢ Eq (Set.image (⇑h) s) (Inter.inter h.toHomeomorph.toPartialHomeomorph.target …
  -/
  simp [h.image_eq_preimage]
  /-
    🎉 no goals
  -/


@[simp]
theorem uniqueMDiffOn_image (h : M ≃ₘ^n⟮I, J⟯ N) (hn : 1 ≤ n) {s : Set M} :
    UniqueMDiffOn J (h '' s) ↔ UniqueMDiffOn I s :=
  ⟨fun hs => h.symm_image_image s ▸ h.symm.uniqueMDiffOn_image_aux hn hs,
    h.uniqueMDiffOn_image_aux hn⟩


@[simp]
theorem uniqueMDiffOn_preimage (h : M ≃ₘ^n⟮I, J⟯ N) (hn : 1 ≤ n) {s : Set N} :
    UniqueMDiffOn I (h ⁻¹' s) ↔ UniqueMDiffOn J s :=
  h.symm_image_eq_preimage s ▸ h.symm.uniqueMDiffOn_image hn

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: should use `E ≃ₘ^n[𝕜] F` notation

@[simp]
theorem uniqueDiffOn_image (h : E ≃ₘ^n⟮𝓘(𝕜, E), 𝓘(𝕜, F)⟯ F) (hn : 1 ≤ n) {s : Set E} :
    UniqueDiffOn 𝕜 (h '' s) ↔ UniqueDiffOn 𝕜 s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_4
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : ENat
    h : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F n
    hn : LE.le 1 n
    s : Set E
    ⊢ Iff (UniqueDiffOn 𝕜 (Set.image (⇑h) s)) (UniqueDiffOn 𝕜 s)
  -/
  simp only [← uniqueMDiffOn_iff_uniqueDiffOn, uniqueMDiffOn_image, hn]
  /-
    🎉 no goals
  -/


@[simp]
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: should use `E ≃ₘ^n[𝕜] F` notation
theorem uniqueDiffOn_preimage (h : E ≃ₘ^n⟮𝓘(𝕜, E), 𝓘(𝕜, F)⟯ F) (hn : 1 ≤ n) {s : Set F} :
    UniqueDiffOn 𝕜 (h ⁻¹' s) ↔ UniqueDiffOn 𝕜 s :=
  h.symm_image_eq_preimage s ▸ h.symm.uniqueDiffOn_image hn


/-- A continuous linear equivalence between normed spaces is a diffeomorphism. -/
def toDiffeomorph : E ≃ₘ[𝕜] E' where
  contMDiff_toFun := e.contDiff.contMDiff
  contMDiff_invFun := e.symm.contDiff.contMDiff
  toEquiv := e.toLinearEquiv.toEquiv


@[simp]
theorem coe_toDiffeomorph : ⇑e.toDiffeomorph = e :=
  rfl


@[simp]
theorem symm_toDiffeomorph : e.symm.toDiffeomorph = e.toDiffeomorph.symm :=
  rfl


@[simp]
theorem coe_toDiffeomorph_symm : ⇑e.toDiffeomorph.symm = e.symm :=
  rfl


/-- Apply a diffeomorphism (e.g., a continuous linear equivalence) to the model vector space. -/
def transDiffeomorph (I : ModelWithCorners 𝕜 E H) (e : E ≃ₘ[𝕜] E') : ModelWithCorners 𝕜 E' H where
  toPartialEquiv := I.toPartialEquiv.trans e.toEquiv.toPartialEquiv
                  /-
                    𝕜 : Type u_1
                    inst✝¹⁸ : NontriviallyNormedField 𝕜
                    E : Type u_2
                    inst✝¹⁷ : NormedAddCommGroup E
                    inst✝¹⁶ : NormedSpace 𝕜 E
                    E' : Type u_3
                    inst✝¹⁵ : NormedAddCommGroup E'
                    inst✝¹⁴ : NormedSpace 𝕜 E'
                    F : Type u_4
                    inst✝¹³ : NormedAddCommGroup F
                    inst✝¹² : NormedSpace 𝕜 F
                    H : Type u_5
                    inst✝¹¹ : TopologicalSpace H
                    H' : Type u_6
                    inst✝¹⁰ : TopologicalSpace H'
                    G : Type u_7
                    inst✝⁹ : TopologicalSpace G
                    G' : Type u_8
                    inst✝⁸ : TopologicalSpace G'
                    I✝ : ModelWithCorners 𝕜 E H
                    I' : ModelWithCorners 𝕜 E' H'
                    J : ModelWithCorners 𝕜 F G
                    J' : ModelWithCorners 𝕜 F G'
                    M : Type u_9
                    inst✝⁷ : TopologicalSpace M
                    inst✝⁶ : ChartedSpace H M
                    M' : Type u_10
                    inst✝⁵ : TopologicalSpace M'
                    inst✝⁴ : ChartedSpace H' M'
                    N : Type u_11
                    inst✝³ : TopologicalSpace N
                    inst✝² : ChartedSpace G N
                    N' : Type u_12
                    inst✝¹ : TopologicalSpace N'
                    inst✝ : ChartedSpace G' N'
                    n : ENat
                    e✝ : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' T …
                    I : ModelWithCorners 𝕜 E H
                    e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' To …
                    ⊢ Eq (I.trans e.toPartialEquiv).source Set.univ
                  -/
  source_eq := by simp
                  /-
                    🎉 no goals
                  -/
                      /-
                        𝕜 : Type u_1
                        inst✝¹⁸ : NontriviallyNormedField 𝕜
                        E : Type u_2
                        inst✝¹⁷ : NormedAddCommGroup E
                        inst✝¹⁶ : NormedSpace 𝕜 E
                        E' : Type u_3
                        inst✝¹⁵ : NormedAddCommGroup E'
                        inst✝¹⁴ : NormedSpace 𝕜 E'
                        F : Type u_4
                        inst✝¹³ : NormedAddCommGroup F
                        inst✝¹² : NormedSpace 𝕜 F
                        H : Type u_5
                        inst✝¹¹ : TopologicalSpace H
                        H' : Type u_6
                        inst✝¹⁰ : TopologicalSpace H'
                        G : Type u_7
                        inst✝⁹ : TopologicalSpace G
                        G' : Type u_8
                        inst✝⁸ : TopologicalSpace G'
                        I✝ : ModelWithCorners 𝕜 E H
                        I' : ModelWithCorners 𝕜 E' H'
                        J : ModelWithCorners 𝕜 F G
                        J' : ModelWithCorners 𝕜 F G'
                        M : Type u_9
                        inst✝⁷ : TopologicalSpace M
                        inst✝⁶ : ChartedSpace H M
                        M' : Type u_10
                        inst✝⁵ : TopologicalSpace M'
                        inst✝⁴ : ChartedSpace H' M'
                        N : Type u_11
                        inst✝³ : TopologicalSpace N
                        inst✝² : ChartedSpace G N
                        N' : Type u_12
                        inst✝¹ : TopologicalSpace N'
                        inst✝ : ChartedSpace G' N'
                        n : ENat
                        e✝ : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' T …
                        I : ModelWithCorners 𝕜 E H
                        e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' To …
                        ⊢ UniqueDiffOn 𝕜 (I.trans e.toPartialEquiv).target
                      -/
  uniqueDiffOn' := by simp [range_comp e, I.uniqueDiffOn]
                      /-
                        🎉 no goals
                      -/
  target_subset_closure_interior := by
    simp only [PartialEquiv.trans_target, Equiv.toPartialEquiv_target,
      Equiv.toPartialEquiv_symm_apply, Diffeomorph.toEquiv_coe_symm, target_eq, univ_inter]
    /-
      𝕜 : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹⁵ : NormedAddCommGroup E'
      inst✝¹⁴ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹³ : NormedAddCommGroup F
      inst✝¹² : NormedSpace 𝕜 F
      H : Type u_5
      inst✝¹¹ : TopologicalSpace H
      H' : Type u_6
      inst✝¹⁰ : TopologicalSpace H'
      G : Type u_7
      inst✝⁹ : TopologicalSpace G
      G' : Type u_8
      inst✝⁸ : TopologicalSpace G'
      I✝ : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      J' : ModelWithCorners 𝕜 F G'
      M : Type u_9
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      M' : Type u_10
      inst✝⁵ : TopologicalSpace M'
      inst✝⁴ : ChartedSpace H' M'
      N : Type u_11
      inst✝³ : TopologicalSpace N
      inst✝² : ChartedSpace G N
      N' : Type u_12
      inst✝¹ : TopologicalSpace N'
      inst✝ : ChartedSpace G' N'
      n : ENat
      e✝ : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' T …
      I : ModelWithCorners 𝕜 E H
      e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' To …
      ⊢ HasSubset.Subset (Set.preimage (⇑e.symm) (Set.range ↑I)) (closure (interior  …
    -/
    change e.toHomeomorph.symm ⁻¹' _ ⊆ closure (interior (e.toHomeomorph.symm ⁻¹' (range I)))
    rw [← e.toHomeomorph.symm.isOpenMap.preimage_interior_eq_interior_preimage
      e.toHomeomorph.continuous_symm,
      ← e.toHomeomorph.symm.isOpenMap.preimage_closure_eq_closure_preimage
      e.toHomeomorph.continuous_symm]
    /-
      𝕜 : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹⁵ : NormedAddCommGroup E'
      inst✝¹⁴ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹³ : NormedAddCommGroup F
      inst✝¹² : NormedSpace 𝕜 F
      H : Type u_5
      inst✝¹¹ : TopologicalSpace H
      H' : Type u_6
      inst✝¹⁰ : TopologicalSpace H'
      G : Type u_7
      inst✝⁹ : TopologicalSpace G
      G' : Type u_8
      inst✝⁸ : TopologicalSpace G'
      I✝ : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      J' : ModelWithCorners 𝕜 F G'
      M : Type u_9
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      M' : Type u_10
      inst✝⁵ : TopologicalSpace M'
      inst✝⁴ : ChartedSpace H' M'
      N : Type u_11
      inst✝³ : TopologicalSpace N
      inst✝² : ChartedSpace G N
      N' : Type u_12
      inst✝¹ : TopologicalSpace N'
      inst✝ : ChartedSpace G' N'
      n : ENat
      e✝ : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' T …
      I : ModelWithCorners 𝕜 E H
      e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' To …
      ⊢ HasSubset.Subset (Set.preimage (⇑e.toHomeomorph.symm) (Set.range ↑I)) (Set.p …
    -/
    exact preimage_mono I.range_subset_closure_interior
    /-
      🎉 no goals
    -/
  continuous_toFun := e.continuous.comp I.continuous
  continuous_invFun := I.continuous_symm.comp e.symm.continuous


@[simp, mfld_simps]
theorem coe_transDiffeomorph : ⇑(I.transDiffeomorph e) = e ∘ I :=
  rfl


@[simp, mfld_simps]
theorem coe_transDiffeomorph_symm : ⇑(I.transDiffeomorph e).symm = I.symm ∘ e.symm :=
  rfl


theorem transDiffeomorph_range : range (I.transDiffeomorph e) = e '' range I :=
  range_comp e I


theorem coe_extChartAt_transDiffeomorph (x : M) :
    ⇑(extChartAt (I.transDiffeomorph e) x) = e ∘ extChartAt I x :=
  rfl


theorem coe_extChartAt_transDiffeomorph_symm (x : M) :
    ⇑(extChartAt (I.transDiffeomorph e) x).symm = (extChartAt I x).symm ∘ e.symm :=
  rfl


theorem extChartAt_transDiffeomorph_target (x : M) :
    (extChartAt (I.transDiffeomorph e) x).target = e.symm ⁻¹' (extChartAt I x).target := by
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝⁴ : NormedAddCommGroup E'
    inst✝³ : NormedSpace 𝕜 E'
    H : Type u_5
    inst✝² : TopologicalSpace H
    I : ModelWithCorners 𝕜 E H
    M : Type u_9
    inst✝¹ : TopologicalSpace M
    inst✝ : ChartedSpace H M
    e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 E') E E' To …
    x : M
    ⊢ Eq (extChartAt (I.transDiffeomorph e) x).target (Set.preimage (⇑e.symm) (ext …
  -/
  simp only [e.range_comp, preimage_preimage, mfld_simps]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


instance smoothManifoldWithCorners_transDiffeomorph [SmoothManifoldWithCorners I M] :
    SmoothManifoldWithCorners (I.transDiffeomorph e) M := by
  /-
    𝕜 : Type u_1
    inst✝¹⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁸ : NormedAddCommGroup E
    inst✝¹⁷ : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹⁶ : NormedAddCommGroup E'
    inst✝¹⁵ : NormedSpace 𝕜 E'
    F : Type u_4
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    H : Type u_5
    inst✝¹² : TopologicalSpace H
    H' : Type u_6
    inst✝¹¹ : TopologicalSpace H'
    G : Type u_7
    inst✝¹⁰ : TopologicalSpace G
    G' : Type u_8
    inst✝⁹ : TopologicalSpace G'
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    J : ModelWithCorners 𝕜 F G
    J' : ModelWithCorners 𝕜 F G'
    M : Type u_9
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    M' : Type u_10
    inst✝⁶ : TopologicalSpace M'
    inst✝⁵ : ChartedSpace H' M'
    N : Type u_11
    inst✝⁴ : TopologicalSpace N
    inst✝³ : ChartedSpace G N
    N' : Type u_12
    inst✝² : TopologicalSpace N'
    inst✝¹ : ChartedSpace G' N'
    n : ENat
    e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F Top. …
    inst✝ : SmoothManifoldWithCorners I M
    ⊢ SmoothManifoldWithCorners (I.transDiffeomorph e) M
  -/
  refine smoothManifoldWithCorners_of_contDiffOn (I.transDiffeomorph e) M fun e₁ e₂ h₁ h₂ => ?_
  refine e.contDiff.comp_contDiffOn
      (((contDiffGroupoid ∞ I).compatible h₁ h₂).1.comp e.symm.contDiff.contDiffOn ?_)
  /-
    𝕜 : Type u_1
    inst✝¹⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁸ : NormedAddCommGroup E
    inst✝¹⁷ : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹⁶ : NormedAddCommGroup E'
    inst✝¹⁵ : NormedSpace 𝕜 E'
    F : Type u_4
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    H : Type u_5
    inst✝¹² : TopologicalSpace H
    H' : Type u_6
    inst✝¹¹ : TopologicalSpace H'
    G : Type u_7
    inst✝¹⁰ : TopologicalSpace G
    G' : Type u_8
    inst✝⁹ : TopologicalSpace G'
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    J : ModelWithCorners 𝕜 F G
    J' : ModelWithCorners 𝕜 F G'
    M : Type u_9
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    M' : Type u_10
    inst✝⁶ : TopologicalSpace M'
    inst✝⁵ : ChartedSpace H' M'
    N : Type u_11
    inst✝⁴ : TopologicalSpace N
    inst✝³ : ChartedSpace G N
    N' : Type u_12
    inst✝² : TopologicalSpace N'
    inst✝¹ : ChartedSpace G' N'
    n : ENat
    e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F Top. …
    inst✝ : SmoothManifoldWithCorners I M
    e₁ e₂ : PartialHomeomorph M H
    h₁ : Membership.mem (atlas H M) e₁
    h₂ : Membership.mem (atlas H M) e₂
    ⊢ Set.MapsTo (↑(e.toPartialEquiv.restr I.target).symm) (Inter.inter (Set.preim …
  -/
  simp only [mapsTo_iff_subset_preimage]
  /-
    𝕜 : Type u_1
    inst✝¹⁹ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝¹⁸ : NormedAddCommGroup E
    inst✝¹⁷ : NormedSpace 𝕜 E
    E' : Type u_3
    inst✝¹⁶ : NormedAddCommGroup E'
    inst✝¹⁵ : NormedSpace 𝕜 E'
    F : Type u_4
    inst✝¹⁴ : NormedAddCommGroup F
    inst✝¹³ : NormedSpace 𝕜 F
    H : Type u_5
    inst✝¹² : TopologicalSpace H
    H' : Type u_6
    inst✝¹¹ : TopologicalSpace H'
    G : Type u_7
    inst✝¹⁰ : TopologicalSpace G
    G' : Type u_8
    inst✝⁹ : TopologicalSpace G'
    I : ModelWithCorners 𝕜 E H
    I' : ModelWithCorners 𝕜 E' H'
    J : ModelWithCorners 𝕜 F G
    J' : ModelWithCorners 𝕜 F G'
    M : Type u_9
    inst✝⁸ : TopologicalSpace M
    inst✝⁷ : ChartedSpace H M
    M' : Type u_10
    inst✝⁶ : TopologicalSpace M'
    inst✝⁵ : ChartedSpace H' M'
    N : Type u_11
    inst✝⁴ : TopologicalSpace N
    inst✝³ : ChartedSpace G N
    N' : Type u_12
    inst✝² : TopologicalSpace N'
    inst✝¹ : ChartedSpace G' N'
    n : ENat
    e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F Top. …
    inst✝ : SmoothManifoldWithCorners I M
    e₁ e₂ : PartialHomeomorph M H
    h₁ : Membership.mem (atlas H M) e₁
    h₂ : Membership.mem (atlas H M) e₂
    ⊢ HasSubset.Subset (Inter.inter (Set.preimage (↑(I.transDiffeomorph e).symm) ( …
  -/
  mfld_set_tac
  /-
    🎉 no goals
  -/


/-- The identity diffeomorphism between a manifold with model `I` and the same manifold
with model `I.trans_diffeomorph e`. -/
def toTransDiffeomorph (e : E ≃ₘ[𝕜] F) : M ≃ₘ⟮I, I.transDiffeomorph e⟯ M where
  toEquiv := Equiv.refl M
  contMDiff_toFun x := by
    /-
      𝕜 : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹⁵ : NormedAddCommGroup E'
      inst✝¹⁴ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹³ : NormedAddCommGroup F
      inst✝¹² : NormedSpace 𝕜 F
      H : Type u_5
      inst✝¹¹ : TopologicalSpace H
      H' : Type u_6
      inst✝¹⁰ : TopologicalSpace H'
      G : Type u_7
      inst✝⁹ : TopologicalSpace G
      G' : Type u_8
      inst✝⁸ : TopologicalSpace G'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      J' : ModelWithCorners 𝕜 F G'
      M : Type u_9
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      M' : Type u_10
      inst✝⁵ : TopologicalSpace M'
      inst✝⁴ : ChartedSpace H' M'
      N : Type u_11
      inst✝³ : TopologicalSpace N
      inst✝² : ChartedSpace G N
      N' : Type u_12
      inst✝¹ : TopologicalSpace N'
      inst✝ : ChartedSpace G' N'
      n : ENat
      e✝ e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F T …
      x : M
      ⊢ ContMDiffAt I (I.transDiffeomorph e) Top.top (⇑(Equiv.refl M)) x
    -/
    refine contMDiffWithinAt_iff'.2 ⟨continuousWithinAt_id, ?_⟩
    /-
      𝕜 : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹⁵ : NormedAddCommGroup E'
      inst✝¹⁴ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹³ : NormedAddCommGroup F
      inst✝¹² : NormedSpace 𝕜 F
      H : Type u_5
      inst✝¹¹ : TopologicalSpace H
      H' : Type u_6
      inst✝¹⁰ : TopologicalSpace H'
      G : Type u_7
      inst✝⁹ : TopologicalSpace G
      G' : Type u_8
      inst✝⁸ : TopologicalSpace G'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      J' : ModelWithCorners 𝕜 F G'
      M : Type u_9
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      M' : Type u_10
      inst✝⁵ : TopologicalSpace M'
      inst✝⁴ : ChartedSpace H' M'
      N : Type u_11
      inst✝³ : TopologicalSpace N
      inst✝² : ChartedSpace G N
      N' : Type u_12
      inst✝¹ : TopologicalSpace N'
      inst✝ : ChartedSpace G' N'
      n : ENat
      e✝ e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F T …
      x : M
      ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑(extChartAt (I.transDiffeomor …
    -/
    refine e.contDiff.contDiffWithinAt.congr_of_mem (fun y hy ↦ ?_) ?_
    · simp only [Equiv.coe_refl, id, (· ∘ ·), I.coe_extChartAt_transDiffeomorph,
        (extChartAt I x).right_inv hy.1]
    · exact
      ⟨(extChartAt I x).map_source (mem_extChartAt_source x), trivial, by simp only [mfld_simps]⟩
  contMDiff_invFun x := by
    /-
      𝕜 : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹⁵ : NormedAddCommGroup E'
      inst✝¹⁴ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹³ : NormedAddCommGroup F
      inst✝¹² : NormedSpace 𝕜 F
      H : Type u_5
      inst✝¹¹ : TopologicalSpace H
      H' : Type u_6
      inst✝¹⁰ : TopologicalSpace H'
      G : Type u_7
      inst✝⁹ : TopologicalSpace G
      G' : Type u_8
      inst✝⁸ : TopologicalSpace G'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      J' : ModelWithCorners 𝕜 F G'
      M : Type u_9
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      M' : Type u_10
      inst✝⁵ : TopologicalSpace M'
      inst✝⁴ : ChartedSpace H' M'
      N : Type u_11
      inst✝³ : TopologicalSpace N
      inst✝² : ChartedSpace G N
      N' : Type u_12
      inst✝¹ : TopologicalSpace N'
      inst✝ : ChartedSpace G' N'
      n : ENat
      e✝ e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F T …
      x : M
      ⊢ ContMDiffAt (I.transDiffeomorph e) I Top.top (⇑(Equiv.refl M).symm) x
    -/
    refine contMDiffWithinAt_iff'.2 ⟨continuousWithinAt_id, ?_⟩
    /-
      𝕜 : Type u_1
      inst✝¹⁸ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝¹⁷ : NormedAddCommGroup E
      inst✝¹⁶ : NormedSpace 𝕜 E
      E' : Type u_3
      inst✝¹⁵ : NormedAddCommGroup E'
      inst✝¹⁴ : NormedSpace 𝕜 E'
      F : Type u_4
      inst✝¹³ : NormedAddCommGroup F
      inst✝¹² : NormedSpace 𝕜 F
      H : Type u_5
      inst✝¹¹ : TopologicalSpace H
      H' : Type u_6
      inst✝¹⁰ : TopologicalSpace H'
      G : Type u_7
      inst✝⁹ : TopologicalSpace G
      G' : Type u_8
      inst✝⁸ : TopologicalSpace G'
      I : ModelWithCorners 𝕜 E H
      I' : ModelWithCorners 𝕜 E' H'
      J : ModelWithCorners 𝕜 F G
      J' : ModelWithCorners 𝕜 F G'
      M : Type u_9
      inst✝⁷ : TopologicalSpace M
      inst✝⁶ : ChartedSpace H M
      M' : Type u_10
      inst✝⁵ : TopologicalSpace M'
      inst✝⁴ : ChartedSpace H' M'
      N : Type u_11
      inst✝³ : TopologicalSpace N
      inst✝² : ChartedSpace G N
      N' : Type u_12
      inst✝¹ : TopologicalSpace N'
      inst✝ : ChartedSpace G' N'
      n : ENat
      e✝ e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F T …
      x : M
      ⊢ ContDiffWithinAt 𝕜 (↑Top.top) (Function.comp (↑(extChartAt I ((Equiv.refl M) …
    -/
    refine e.symm.contDiff.contDiffWithinAt.congr_of_mem (fun y hy => ?_) ?_
      /-
        case refine_1
        𝕜 : Type u_1
        inst✝¹⁸ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝¹⁷ : NormedAddCommGroup E
        inst✝¹⁶ : NormedSpace 𝕜 E
        E' : Type u_3
        inst✝¹⁵ : NormedAddCommGroup E'
        inst✝¹⁴ : NormedSpace 𝕜 E'
        F : Type u_4
        inst✝¹³ : NormedAddCommGroup F
        inst✝¹² : NormedSpace 𝕜 F
        H : Type u_5
        inst✝¹¹ : TopologicalSpace H
        H' : Type u_6
        inst✝¹⁰ : TopologicalSpace H'
        G : Type u_7
        inst✝⁹ : TopologicalSpace G
        G' : Type u_8
        inst✝⁸ : TopologicalSpace G'
        I : ModelWithCorners 𝕜 E H
        I' : ModelWithCorners 𝕜 E' H'
        J : ModelWithCorners 𝕜 F G
        J' : ModelWithCorners 𝕜 F G'
        M : Type u_9
        inst✝⁷ : TopologicalSpace M
        inst✝⁶ : ChartedSpace H M
        M' : Type u_10
        inst✝⁵ : TopologicalSpace M'
        inst✝⁴ : ChartedSpace H' M'
        N : Type u_11
        inst✝³ : TopologicalSpace N
        inst✝² : ChartedSpace G N
        N' : Type u_12
        inst✝¹ : TopologicalSpace N'
        inst✝ : ChartedSpace G' N'
        n : ENat
        e✝ e : Diffeomorph (modelWithCornersSelf 𝕜 E) (modelWithCornersSelf 𝕜 F) E F T …
        x : M
        y : F
        hy : Membership.mem (Inter.inter (extChartAt (I.transDiffeomorph e) x).target  …
        ⊢ Eq (Function.comp (↑(extChartAt I ((Equiv.refl M).symm x))) (Function.comp ⇑ …
      -/
    · simp only [mem_inter_iff, I.extChartAt_transDiffeomorph_target] at hy
      simp only [Equiv.coe_refl, Equiv.refl_symm, id, (· ∘ ·),
        I.coe_extChartAt_transDiffeomorph_symm, (extChartAt I x).right_inv hy.1]
    exact ⟨(extChartAt _ x).map_source (mem_extChartAt_source x), trivial, by
      simp only [e.symm_apply_apply, Equiv.refl_symm, Equiv.coe_refl, mfld_simps]⟩


@[simp]
theorem contMDiffWithinAt_transDiffeomorph_right {f : M' → M} {x s} :
    ContMDiffWithinAt I' (I.transDiffeomorph e) n f s x ↔ ContMDiffWithinAt I' I n f s x :=
  (toTransDiffeomorph I M e).contMDiffWithinAt_diffeomorph_comp_iff le_top


@[simp]
theorem contMDiffAt_transDiffeomorph_right {f : M' → M} {x} :
    ContMDiffAt I' (I.transDiffeomorph e) n f x ↔ ContMDiffAt I' I n f x :=
  (toTransDiffeomorph I M e).contMDiffAt_diffeomorph_comp_iff le_top


@[simp]
theorem contMDiffOn_transDiffeomorph_right {f : M' → M} {s} :
    ContMDiffOn I' (I.transDiffeomorph e) n f s ↔ ContMDiffOn I' I n f s :=
  (toTransDiffeomorph I M e).contMDiffOn_diffeomorph_comp_iff le_top


@[simp]
theorem contMDiff_transDiffeomorph_right {f : M' → M} :
    ContMDiff I' (I.transDiffeomorph e) n f ↔ ContMDiff I' I n f :=
  (toTransDiffeomorph I M e).contMDiff_diffeomorph_comp_iff le_top


@[deprecated (since := "2024-11-21")]
alias smooth_transDiffeomorph_right := contMDiff_transDiffeomorph_right


@[simp]
theorem contMDiffWithinAt_transDiffeomorph_left {f : M → M'} {x s} :
    ContMDiffWithinAt (I.transDiffeomorph e) I' n f s x ↔ ContMDiffWithinAt I I' n f s x :=
  ((toTransDiffeomorph I M e).contMDiffWithinAt_comp_diffeomorph_iff le_top).symm


@[simp]
theorem contMDiffAt_transDiffeomorph_left {f : M → M'} {x} :
    ContMDiffAt (I.transDiffeomorph e) I' n f x ↔ ContMDiffAt I I' n f x :=
  ((toTransDiffeomorph I M e).contMDiffAt_comp_diffeomorph_iff le_top).symm


@[simp]
theorem contMDiffOn_transDiffeomorph_left {f : M → M'} {s} :
    ContMDiffOn (I.transDiffeomorph e) I' n f s ↔ ContMDiffOn I I' n f s :=
  ((toTransDiffeomorph I M e).contMDiffOn_comp_diffeomorph_iff le_top).symm


@[simp]
theorem contMDiff_transDiffeomorph_left {f : M → M'} :
    ContMDiff (I.transDiffeomorph e) I' n f ↔ ContMDiff I I' n f :=
  ((toTransDiffeomorph I M e).contMDiff_comp_diffeomorph_iff le_top).symm


@[deprecated (since := "2024-11-21")]
alias smooth_transDiffeomorph_left := contMDiff_transDiffeomorph_left


