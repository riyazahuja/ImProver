include p in
theorem IsLocalization.flat : Module.Flat R S :=
  (Module.Flat.iff_lTensor_injective' _ _).mpr fun I ↦ by
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      p : Submonoid R
      inst✝ : IsLocalization p S
      I : Ideal R
      ⊢ Function.Injective ⇑(LinearMap.lTensor S (Submodule.subtype I))
    -/
    have h := (I.isLocalizedModule S p (Algebra.linearMap R S)).isBaseChange _ S _
    have : I.subtype.lTensor S = (TensorProduct.rid R S).symm.comp
        ((Submodule.subtype _ ∘ₗ h.equiv.toLinearMap).restrictScalars R) := by
      rw [LinearEquiv.eq_toLinearMap_symm_comp]; ext
      simp [h.equiv_tmul, Algebra.smul_def, mul_comm, Algebra.ofId_apply]
    /-
      R : Type u_1
      S : Type u_2
      inst✝³ : CommRing R
      inst✝² : CommRing S
      inst✝¹ : Algebra R S
      p : Submonoid R
      inst✝ : IsLocalization p S
      I : Ideal R
      h : IsBaseChange S (Submodule.toLocalized' S p (Algebra.linearMap R S) I)
      this : Eq (LinearMap.lTensor S (Submodule.subtype I)) ((↑(TensorProduct.rid R  …
      ⊢ Function.Injective ⇑(LinearMap.lTensor S (Submodule.subtype I))
    -/
    simpa [this, - Subtype.val_injective] using Subtype.val_injective
    /-
      🎉 no goals
    -/


instance Localization.flat : Module.Flat R (Localization p) := IsLocalization.flat _ p


include p in
theorem flat_iff_of_isLocalization : Flat S M ↔ Flat R M :=
  have := isLocalizedModule_id p M S
  have := IsLocalization.flat S p
  ⟨fun _ ↦ .trans R S M, fun _ ↦ .of_isLocalizedModule S p .id⟩


include f in
theorem flat_of_isLocalized_maximal (H : ∀ (P : Ideal S) [P.IsMaximal], Flat R (Mₚ P)) :
    Module.Flat R M := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    Mₚ : (P : Ideal S) → [inst : P.IsMaximal] → Type u_4
    inst✝⁴ : (P : Ideal S) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
    inst✝³ : (P : Ideal S) → [inst : P.IsMaximal] → Module R (Mₚ P)
    inst✝² : (P : Ideal S) → [inst : P.IsMaximal] → Module S (Mₚ P)
    inst✝¹ : ∀ (P : Ideal S) [inst : P.IsMaximal], IsScalarTower R S (Mₚ P)
    f : (P : Ideal S) → [inst : P.IsMaximal] → LinearMap (RingHom.id S) M (Mₚ P)
    inst✝ : ∀ (P : Ideal S) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    H : ∀ (P : Ideal S) [inst : P.IsMaximal], Module.Flat R (Mₚ P)
    ⊢ Module.Flat R M
  -/
  simp_rw [Flat.iff_lTensor_injective'] at H ⊢
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    Mₚ : (P : Ideal S) → [inst : P.IsMaximal] → Type u_4
    inst✝⁴ : (P : Ideal S) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
    inst✝³ : (P : Ideal S) → [inst : P.IsMaximal] → Module R (Mₚ P)
    inst✝² : (P : Ideal S) → [inst : P.IsMaximal] → Module S (Mₚ P)
    inst✝¹ : ∀ (P : Ideal S) [inst : P.IsMaximal], IsScalarTower R S (Mₚ P)
    f : (P : Ideal S) → [inst : P.IsMaximal] → LinearMap (RingHom.id S) M (Mₚ P)
    inst✝ : ∀ (P : Ideal S) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    H : ∀ (P : Ideal S) [inst : P.IsMaximal] (I : Ideal R), Function.Injective ⇑(L …
    ⊢ ∀ (I : Ideal R), Function.Injective ⇑(LinearMap.lTensor M (Submodule.subtype …
  -/
  simp_rw [← AlgebraTensorModule.coe_lTensor (A := S)]
  refine fun I ↦ injective_of_isLocalized_maximal _ (fun P ↦ AlgebraTensorModule.rTensor R _ (f P))
    _ (fun P ↦ AlgebraTensorModule.rTensor R _ (f P)) _ fun P hP ↦ ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    Mₚ : (P : Ideal S) → [inst : P.IsMaximal] → Type u_4
    inst✝⁴ : (P : Ideal S) → [inst : P.IsMaximal] → AddCommGroup (Mₚ P)
    inst✝³ : (P : Ideal S) → [inst : P.IsMaximal] → Module R (Mₚ P)
    inst✝² : (P : Ideal S) → [inst : P.IsMaximal] → Module S (Mₚ P)
    inst✝¹ : ∀ (P : Ideal S) [inst : P.IsMaximal], IsScalarTower R S (Mₚ P)
    f : (P : Ideal S) → [inst : P.IsMaximal] → LinearMap (RingHom.id S) M (Mₚ P)
    inst✝ : ∀ (P : Ideal S) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    H : ∀ (P : Ideal S) [inst : P.IsMaximal] (I : Ideal R), Function.Injective ⇑(L …
    I : Ideal R
    P : Ideal S
    hP : P.IsMaximal
    ⊢ Function.Injective ⇑((IsLocalizedModule.map P.primeCompl ((fun P [P.IsMaxima …
  -/
  simpa [IsLocalizedModule.map_lTensor] using H P I
  /-
    🎉 no goals
  -/


theorem flat_of_localized_maximal
    (h : ∀ (P : Ideal R) [P.IsMaximal], Flat R (LocalizedModule P.primeCompl M)) :
    Flat R M :=
  flat_of_isLocalized_maximal _ _ _ (fun _ _ ↦ mkLinearMap _ _) h


include g in
theorem flat_of_isLocalized_span (H : ∀ r : s, Module.Flat R (Mₛ r)) :
    Module.Flat R M := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    s : Set S
    spn : Eq (Ideal.span s) Top.top
    Mₛ : ↑s → Type u_5
    inst✝⁴ : (r : ↑s) → AddCommGroup (Mₛ r)
    inst✝³ : (r : ↑s) → Module R (Mₛ r)
    inst✝² : (r : ↑s) → Module S (Mₛ r)
    inst✝¹ : ∀ (r : ↑s), IsScalarTower R S (Mₛ r)
    g : (r : ↑s) → LinearMap (RingHom.id S) M (Mₛ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    H : ∀ (r : ↑s), Module.Flat R (Mₛ r)
    ⊢ Module.Flat R M
  -/
  simp_rw [Flat.iff_lTensor_injective'] at H ⊢
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    s : Set S
    spn : Eq (Ideal.span s) Top.top
    Mₛ : ↑s → Type u_5
    inst✝⁴ : (r : ↑s) → AddCommGroup (Mₛ r)
    inst✝³ : (r : ↑s) → Module R (Mₛ r)
    inst✝² : (r : ↑s) → Module S (Mₛ r)
    inst✝¹ : ∀ (r : ↑s), IsScalarTower R S (Mₛ r)
    g : (r : ↑s) → LinearMap (RingHom.id S) M (Mₛ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    H : ∀ (r : ↑s) (I : Ideal R), Function.Injective ⇑(LinearMap.lTensor (Mₛ r) (S …
    ⊢ ∀ (I : Ideal R), Function.Injective ⇑(LinearMap.lTensor M (Submodule.subtype …
  -/
  simp_rw [← AlgebraTensorModule.coe_lTensor (A := S)]
  refine fun I ↦ injective_of_isLocalized_span s spn _ (fun r ↦ AlgebraTensorModule.rTensor
    R _ (g r)) _ (fun r ↦ AlgebraTensorModule.rTensor R _ (g r)) _ fun r ↦ ?_
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    M : Type u_3
    inst✝⁸ : AddCommGroup M
    inst✝⁷ : Module R M
    inst✝⁶ : Module S M
    inst✝⁵ : IsScalarTower R S M
    s : Set S
    spn : Eq (Ideal.span s) Top.top
    Mₛ : ↑s → Type u_5
    inst✝⁴ : (r : ↑s) → AddCommGroup (Mₛ r)
    inst✝³ : (r : ↑s) → Module R (Mₛ r)
    inst✝² : (r : ↑s) → Module S (Mₛ r)
    inst✝¹ : ∀ (r : ↑s), IsScalarTower R S (Mₛ r)
    g : (r : ↑s) → LinearMap (RingHom.id S) M (Mₛ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    H : ∀ (r : ↑s) (I : Ideal R), Function.Injective ⇑(LinearMap.lTensor (Mₛ r) (S …
    I : Ideal R
    r : ↑s
    ⊢ Function.Injective ⇑((IsLocalizedModule.map (Submonoid.powers ↑r) ((fun r => …
  -/
  simpa [IsLocalizedModule.map_lTensor] using H r I
  /-
    🎉 no goals
  -/


theorem flat_of_localized_span
    (h : ∀ r : s, Flat S (LocalizedModule (.powers r.1) M)) :
    Flat S M :=
  flat_of_isLocalized_span _ _ _ spn _ (fun _ ↦ mkLinearMap _ _) h


