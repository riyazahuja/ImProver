local notation "k" => ResidueField R

local notation "𝔪" => maximalIdeal R


theorem map_mkQ_eq {N₁ N₂ : Submodule R M} (h : N₁ ≤ N₂) (h' : N₂.FG) :
    N₁.map (Submodule.mkQ (𝔪 • N₂)) = N₂.map (Submodule.mkQ (𝔪 • N₂)) ↔ N₁ = N₂ := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    inst✝ : IsLocalRing R
    N₁ N₂ : Submodule R M
    h : LE.le N₁ N₂
    h' : N₂.FG
    ⊢ Iff (Eq (Submodule.map (HSMul.hSMul (IsLocalRing.maximalIdeal R) N₂).mkQ N₁) …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsLocalRing R
      N₁ N₂ : Submodule R M
      h : LE.le N₁ N₂
      h' : N₂.FG
      ⊢ Eq (Submodule.map (HSMul.hSMul (IsLocalRing.maximalIdeal R) N₂).mkQ N₁) (Sub …
    -/
  · intro hN
    have : N₂ ≤ 𝔪 • N₂ ⊔ N₁ := by
      simpa using Submodule.comap_mono (f := Submodule.mkQ (𝔪 • N₂)) hN.ge
    /-
      case mp
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsLocalRing R
      N₁ N₂ : Submodule R M
      h : LE.le N₁ N₂
      h' : N₂.FG
      hN : Eq (Submodule.map (HSMul.hSMul (IsLocalRing.maximalIdeal R) N₂).mkQ N₁) ( …
      this : LE.le N₂ (Max.max (HSMul.hSMul (IsLocalRing.maximalIdeal R) N₂) N₁)
      ⊢ Eq N₁ N₂
    -/
    rw [sup_comm] at this
    exact h.antisymm (Submodule.le_of_le_smul_of_le_jacobson_bot h'
      (by rw [jacobson_eq_maximalIdeal]; exact bot_ne_top) this)
    /-
      case mpr
      R : Type u_1
      inst✝³ : CommRing R
      M : Type u_2
      inst✝² : AddCommGroup M
      inst✝¹ : Module R M
      inst✝ : IsLocalRing R
      N₁ N₂ : Submodule R M
      h : LE.le N₁ N₂
      h' : N₂.FG
      ⊢ Eq N₁ N₂ → Eq (Submodule.map (HSMul.hSMul (IsLocalRing.maximalIdeal R) N₂).m …
    -/
  · rintro rfl; simp
                /-
                  🎉 no goals
                -/


theorem map_mkQ_eq_top {N : Submodule R M} [Module.Finite R M] :
    N.map (Submodule.mkQ (𝔪 • ⊤)) = ⊤ ↔ N = ⊤ := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    N : Submodule R M
    inst✝ : Module.Finite R M
    ⊢ Iff (Eq (Submodule.map (HSMul.hSMul (IsLocalRing.maximalIdeal R) Top.top).mk …
  -/
  rw [← map_mkQ_eq (N₁ := N) le_top Module.Finite.out, Submodule.map_top, Submodule.range_mkQ]
  /-
    🎉 no goals
  -/


theorem map_tensorProduct_mk_eq_top {N : Submodule R M} [Module.Finite R M] :
    N.map (TensorProduct.mk R k M 1) = ⊤ ↔ N = ⊤ := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    N : Submodule R M
    inst✝ : Module.Finite R M
    ⊢ Iff (Eq (Submodule.map ((TensorProduct.mk R (IsLocalRing.ResidueField R) M)  …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsLocalRing R
      N : Submodule R M
      inst✝ : Module.Finite R M
      ⊢ Eq (Submodule.map ((TensorProduct.mk R (IsLocalRing.ResidueField R) M) 1) N) …
    -/
  · intro hN
    letI : Module k (M ⧸ (𝔪 • ⊤ : Submodule R M)) :=
      inferInstanceAs (Module (R ⧸ 𝔪) (M ⧸ 𝔪 • (⊤ : Submodule R M)))
    letI : IsScalarTower R k (M ⧸ (𝔪 • ⊤ : Submodule R M)) :=
      inferInstanceAs (IsScalarTower R (R ⧸ 𝔪) (M ⧸ 𝔪 • (⊤ : Submodule R M)))
    let f := AlgebraTensorModule.lift (((LinearMap.ringLmapEquivSelf k k _).symm
      (Submodule.mkQ (𝔪 • ⊤ : Submodule R M))).restrictScalars R)
    /-
      case mp
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsLocalRing R
      N : Submodule R M
      inst✝ : Module.Finite R M
      hN : Eq (Submodule.map ((TensorProduct.mk R (IsLocalRing.ResidueField R) M) 1) …
      this✝ : Module (IsLocalRing.ResidueField R) (HasQuotient.Quotient M (HSMul.hSM …
      this : IsScalarTower R (IsLocalRing.ResidueField R) (HasQuotient.Quotient M (H …
      f : LinearMap (RingHom.id R) (TensorProduct R (IsLocalRing.ResidueField R) M)  …
      ⊢ Eq N Top.top
    -/
    have : f.comp (TensorProduct.mk R k M 1) = Submodule.mkQ (𝔪 • ⊤) := by ext; simp [f]
    have hf : Function.Surjective f := by
      intro x; obtain ⟨x, rfl⟩ := Submodule.mkQ_surjective _ x
      rw [← this, LinearMap.comp_apply]; exact ⟨_, rfl⟩
    /-
      case mp
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsLocalRing R
      N : Submodule R M
      inst✝ : Module.Finite R M
      hN : Eq (Submodule.map ((TensorProduct.mk R (IsLocalRing.ResidueField R) M) 1) …
      this✝¹ : Module (IsLocalRing.ResidueField R) (HasQuotient.Quotient M (HSMul.hS …
      this✝ : IsScalarTower R (IsLocalRing.ResidueField R) (HasQuotient.Quotient M ( …
      f : LinearMap (RingHom.id R) (TensorProduct R (IsLocalRing.ResidueField R) M)  …
      this : Eq (f.comp ((TensorProduct.mk R (IsLocalRing.ResidueField R) M) 1)) (HS …
      hf : Function.Surjective ⇑f
      ⊢ Eq N Top.top
    -/
    apply_fun Submodule.map f at hN
    rwa [← Submodule.map_comp, this, Submodule.map_top, LinearMap.range_eq_top.2 hf,
      map_mkQ_eq_top] at hN
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsLocalRing R
      N : Submodule R M
      inst✝ : Module.Finite R M
      ⊢ Eq N Top.top → Eq (Submodule.map ((TensorProduct.mk R (IsLocalRing.ResidueFi …
    -/
  · rintro rfl; rw [Submodule.map_top, LinearMap.range_eq_top]
    /-
      case mpr
      R : Type u_1
      inst✝⁴ : CommRing R
      M : Type u_2
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : IsLocalRing R
      inst✝ : Module.Finite R M
      ⊢ Function.Surjective ⇑((TensorProduct.mk R (IsLocalRing.ResidueField R) M) 1)
    -/
    exact TensorProduct.mk_surjective R M k Ideal.Quotient.mk_surjective
    /-
      🎉 no goals
    -/


theorem subsingleton_tensorProduct [Module.Finite R M] :
    Subsingleton (k ⊗[R] M) ↔ Subsingleton M := by
  rw [← Submodule.subsingleton_iff R, ← subsingleton_iff_bot_eq_top,
    ← Submodule.subsingleton_iff R, ← subsingleton_iff_bot_eq_top,
    ← map_tensorProduct_mk_eq_top (M := M), Submodule.map_bot]


theorem span_eq_top_of_tmul_eq_basis [Module.Finite R M] {ι}
    (f : ι → M) (b : Basis ι k (k ⊗[R] M))
    (hb : ∀ i, 1 ⊗ₜ f i = b i) : Submodule.span R (Set.range f) = ⊤ := by
  rw [← map_tensorProduct_mk_eq_top, Submodule.map_span, ← Submodule.restrictScalars_span R k
    Ideal.Quotient.mk_surjective, Submodule.restrictScalars_eq_top_iff,
    ← b.span_eq, ← Set.range_comp]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.Finite R M
    ι : Type u_3
    f : ι → M
    b : Basis ι (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing.Residue …
    hb : ∀ (i : ι), Eq (TensorProduct.tmul R 1 (f i)) (b i)
    ⊢ Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.comp (⇑ …
  -/
  simp only [Function.comp_def, mk_apply, hb, Basis.span_eq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-11")]
alias LocalRing.map_mkQ_eq := IsLocalRing.map_mkQ_eq


@[deprecated (since := "2024-11-11")]
alias LocalRing.map_mkQ_eq_top := IsLocalRing.map_mkQ_eq_top


@[deprecated (since := "2024-11-11")]
alias LocalRing.map_tensorProduct_mk_eq_top := IsLocalRing.map_tensorProduct_mk_eq_top


@[deprecated (since := "2024-11-11")]
alias LocalRing.subsingleton_tensorProduct := IsLocalRing.subsingleton_tensorProduct


@[deprecated (since := "2024-11-11")]
alias LocalRing.span_eq_top_of_tmul_eq_basis := IsLocalRing.span_eq_top_of_tmul_eq_basis


open Function in
/--
Given `M₁ → M₂ → M₃ → 0` and `N₁ → N₂ → N₃ → 0`,
if `M₁ ⊗ N₃ → M₂ ⊗ N₃` and `M₂ ⊗ N₁ → M₂ ⊗ N₂` are both injective,
then `M₃ ⊗ N₁ → M₃ ⊗ N₂` is also injective.
-/
theorem lTensor_injective_of_exact_of_exact_of_rTensor_injective
    {M₁ M₂ M₃ N₁ N₂ N₃}
    [AddCommGroup M₁] [Module R M₁] [AddCommGroup M₂] [Module R M₂] [AddCommGroup M₃] [Module R M₃]
    [AddCommGroup N₁] [Module R N₁] [AddCommGroup N₂] [Module R N₂] [AddCommGroup N₃] [Module R N₃]
    {f₁ : M₁ →ₗ[R] M₂} {f₂ : M₂ →ₗ[R] M₃} {g₁ : N₁ →ₗ[R] N₂} {g₂ : N₂ →ₗ[R] N₃}
    (hfexact : Exact f₁ f₂) (hfsurj : Surjective f₂)
    (hgexact : Exact g₁ g₂) (hgsurj : Surjective g₂)
    (hfinj : Injective (f₁.rTensor N₃)) (hginj : Injective (g₁.lTensor M₂)) :
    Injective (g₁.lTensor M₃) := by
  /-
    R : Type u_7
    inst✝¹² : CommRing R
    M₁ : Type u_1
    M₂ : Type u_2
    M₃ : Type u_3
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_6
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : AddCommGroup M₃
    inst✝⁶ : Module R M₃
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup N₂
    inst✝² : Module R N₂
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hfexact : Function.Exact ⇑f₁ ⇑f₂
    hfsurj : Function.Surjective ⇑f₂
    hgexact : Function.Exact ⇑g₁ ⇑g₂
    hgsurj : Function.Surjective ⇑g₂
    hfinj : Function.Injective ⇑(LinearMap.rTensor N₃ f₁)
    hginj : Function.Injective ⇑(LinearMap.lTensor M₂ g₁)
    ⊢ Function.Injective ⇑(LinearMap.lTensor M₃ g₁)
  -/
  rw [injective_iff_map_eq_zero]
  /-
    R : Type u_7
    inst✝¹² : CommRing R
    M₁ : Type u_1
    M₂ : Type u_2
    M₃ : Type u_3
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_6
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : AddCommGroup M₃
    inst✝⁶ : Module R M₃
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup N₂
    inst✝² : Module R N₂
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hfexact : Function.Exact ⇑f₁ ⇑f₂
    hfsurj : Function.Surjective ⇑f₂
    hgexact : Function.Exact ⇑g₁ ⇑g₂
    hgsurj : Function.Surjective ⇑g₂
    hfinj : Function.Injective ⇑(LinearMap.rTensor N₃ f₁)
    hginj : Function.Injective ⇑(LinearMap.lTensor M₂ g₁)
    ⊢ ∀ (a : TensorProduct R M₃ N₁), Eq ((LinearMap.lTensor M₃ g₁) a) 0 → Eq a 0
  -/
  intro x hx
  /-
    R : Type u_7
    inst✝¹² : CommRing R
    M₁ : Type u_1
    M₂ : Type u_2
    M₃ : Type u_3
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_6
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : AddCommGroup M₃
    inst✝⁶ : Module R M₃
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup N₂
    inst✝² : Module R N₂
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hfexact : Function.Exact ⇑f₁ ⇑f₂
    hfsurj : Function.Surjective ⇑f₂
    hgexact : Function.Exact ⇑g₁ ⇑g₂
    hgsurj : Function.Surjective ⇑g₂
    hfinj : Function.Injective ⇑(LinearMap.rTensor N₃ f₁)
    hginj : Function.Injective ⇑(LinearMap.lTensor M₂ g₁)
    x : TensorProduct R M₃ N₁
    hx : Eq ((LinearMap.lTensor M₃ g₁) x) 0
    ⊢ Eq x 0
  -/
  obtain ⟨x, rfl⟩ := f₂.rTensor_surjective N₁ hfsurj x
  have : f₂.rTensor N₂ (g₁.lTensor M₂ x) = 0 := by
    rw [← hx, ← LinearMap.comp_apply, ← LinearMap.comp_apply, LinearMap.rTensor_comp_lTensor,
      LinearMap.lTensor_comp_rTensor]
  /-
    case intro
    R : Type u_7
    inst✝¹² : CommRing R
    M₁ : Type u_1
    M₂ : Type u_2
    M₃ : Type u_3
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_6
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : AddCommGroup M₃
    inst✝⁶ : Module R M₃
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup N₂
    inst✝² : Module R N₂
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hfexact : Function.Exact ⇑f₁ ⇑f₂
    hfsurj : Function.Surjective ⇑f₂
    hgexact : Function.Exact ⇑g₁ ⇑g₂
    hgsurj : Function.Surjective ⇑g₂
    hfinj : Function.Injective ⇑(LinearMap.rTensor N₃ f₁)
    hginj : Function.Injective ⇑(LinearMap.lTensor M₂ g₁)
    x : TensorProduct R M₂ N₁
    hx : Eq ((LinearMap.lTensor M₃ g₁) ((LinearMap.rTensor N₁ f₂) x)) 0
    this : Eq ((LinearMap.rTensor N₂ f₂) ((LinearMap.lTensor M₂ g₁) x)) 0
    ⊢ Eq ((LinearMap.rTensor N₁ f₂) x) 0
  -/
  obtain ⟨y, hy⟩ := (rTensor_exact N₂ hfexact hfsurj _).mp this
  have : g₂.lTensor M₁ y = 0 := by
    apply hfinj
    trans g₂.lTensor M₂ (g₁.lTensor M₂ x)
    · rw [← hy, ← LinearMap.comp_apply, ← LinearMap.comp_apply, LinearMap.rTensor_comp_lTensor,
        LinearMap.lTensor_comp_rTensor]
    rw [← LinearMap.comp_apply, ← LinearMap.lTensor_comp, hgexact.linearMap_comp_eq_zero]
    simp
  /-
    case intro.intro
    R : Type u_7
    inst✝¹² : CommRing R
    M₁ : Type u_1
    M₂ : Type u_2
    M₃ : Type u_3
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_6
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : AddCommGroup M₃
    inst✝⁶ : Module R M₃
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup N₂
    inst✝² : Module R N₂
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hfexact : Function.Exact ⇑f₁ ⇑f₂
    hfsurj : Function.Surjective ⇑f₂
    hgexact : Function.Exact ⇑g₁ ⇑g₂
    hgsurj : Function.Surjective ⇑g₂
    hfinj : Function.Injective ⇑(LinearMap.rTensor N₃ f₁)
    hginj : Function.Injective ⇑(LinearMap.lTensor M₂ g₁)
    x : TensorProduct R M₂ N₁
    hx : Eq ((LinearMap.lTensor M₃ g₁) ((LinearMap.rTensor N₁ f₂) x)) 0
    this✝ : Eq ((LinearMap.rTensor N₂ f₂) ((LinearMap.lTensor M₂ g₁) x)) 0
    y : TensorProduct R M₁ N₂
    hy : Eq ((LinearMap.rTensor N₂ f₁) y) ((LinearMap.lTensor M₂ g₁) x)
    this : Eq ((LinearMap.lTensor M₁ g₂) y) 0
    ⊢ Eq ((LinearMap.rTensor N₁ f₂) x) 0
  -/
  obtain ⟨z, rfl⟩ := (lTensor_exact _ hgexact hgsurj _).mp this
  obtain rfl : f₁.rTensor N₁ z = x := by
    apply hginj
    simp only [← hy, ← LinearMap.comp_apply, ← LinearMap.comp_apply, LinearMap.lTensor_comp_rTensor,
      LinearMap.rTensor_comp_lTensor]
  /-
    case intro.intro.intro
    R : Type u_7
    inst✝¹² : CommRing R
    M₁ : Type u_1
    M₂ : Type u_2
    M₃ : Type u_3
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_6
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : AddCommGroup M₃
    inst✝⁶ : Module R M₃
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup N₂
    inst✝² : Module R N₂
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hfexact : Function.Exact ⇑f₁ ⇑f₂
    hfsurj : Function.Surjective ⇑f₂
    hgexact : Function.Exact ⇑g₁ ⇑g₂
    hgsurj : Function.Surjective ⇑g₂
    hfinj : Function.Injective ⇑(LinearMap.rTensor N₃ f₁)
    hginj : Function.Injective ⇑(LinearMap.lTensor M₂ g₁)
    z : TensorProduct R M₁ N₁
    this✝ : Eq ((LinearMap.lTensor M₁ g₂) ((LinearMap.lTensor M₁ g₁) z)) 0
    hx : Eq ((LinearMap.lTensor M₃ g₁) ((LinearMap.rTensor N₁ f₂) ((LinearMap.rTen …
    this : Eq ((LinearMap.rTensor N₂ f₂) ((LinearMap.lTensor M₂ g₁) ((LinearMap.rT …
    hy : Eq ((LinearMap.rTensor N₂ f₁) ((LinearMap.lTensor M₁ g₁) z)) ((LinearMap. …
    ⊢ Eq ((LinearMap.rTensor N₁ f₂) ((LinearMap.rTensor N₁ f₁) z)) 0
  -/
  rw [← LinearMap.comp_apply, ← LinearMap.rTensor_comp, hfexact.linearMap_comp_eq_zero]
  /-
    case intro.intro.intro
    R : Type u_7
    inst✝¹² : CommRing R
    M₁ : Type u_1
    M₂ : Type u_2
    M₃ : Type u_3
    N₁ : Type u_4
    N₂ : Type u_5
    N₃ : Type u_6
    inst✝¹¹ : AddCommGroup M₁
    inst✝¹⁰ : Module R M₁
    inst✝⁹ : AddCommGroup M₂
    inst✝⁸ : Module R M₂
    inst✝⁷ : AddCommGroup M₃
    inst✝⁶ : Module R M₃
    inst✝⁵ : AddCommGroup N₁
    inst✝⁴ : Module R N₁
    inst✝³ : AddCommGroup N₂
    inst✝² : Module R N₂
    inst✝¹ : AddCommGroup N₃
    inst✝ : Module R N₃
    f₁ : LinearMap (RingHom.id R) M₁ M₂
    f₂ : LinearMap (RingHom.id R) M₂ M₃
    g₁ : LinearMap (RingHom.id R) N₁ N₂
    g₂ : LinearMap (RingHom.id R) N₂ N₃
    hfexact : Function.Exact ⇑f₁ ⇑f₂
    hfsurj : Function.Surjective ⇑f₂
    hgexact : Function.Exact ⇑g₁ ⇑g₂
    hgsurj : Function.Surjective ⇑g₂
    hfinj : Function.Injective ⇑(LinearMap.rTensor N₃ f₁)
    hginj : Function.Injective ⇑(LinearMap.lTensor M₂ g₁)
    z : TensorProduct R M₁ N₁
    this✝ : Eq ((LinearMap.lTensor M₁ g₂) ((LinearMap.lTensor M₁ g₁) z)) 0
    hx : Eq ((LinearMap.lTensor M₃ g₁) ((LinearMap.rTensor N₁ f₂) ((LinearMap.rTen …
    this : Eq ((LinearMap.rTensor N₂ f₂) ((LinearMap.lTensor M₂ g₁) ((LinearMap.rT …
    hy : Eq ((LinearMap.rTensor N₂ f₁) ((LinearMap.lTensor M₁ g₁) z)) ((LinearMap. …
    ⊢ Eq ((LinearMap.rTensor N₁ 0) z) 0
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `M` is of finite presentation over a local ring `(R, 𝔪, k)` such that
`𝔪 ⊗ M → M` is injective, then every family of elements that is a `k`-basis of
`k ⊗ M` is an `R`-basis of `M`. -/
lemma exists_basis_of_basis_baseChange [Module.FinitePresentation R M]
    {ι : Type u} (v : ι → M) (hli : LinearIndependent k (TensorProduct.mk R k M 1 ∘ v))
    (hsp : Submodule.span k (Set.range (TensorProduct.mk R k M 1 ∘ v)) = ⊤)
    (H : Function.Injective ((𝔪).subtype.rTensor M)) :
    ∃ (b : Basis ι R M), ∀ i, b i = v i := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    ι : Type u
    v : ι → M
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (⇑((Tensor …
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    ⊢ Exists fun b => ∀ (i : ι), Eq (b i) (v i)
  -/
  let bk : Basis ι k (k ⊗[R] M) := Basis.mk hli (by rw [hsp])
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    ι : Type u
    v : ι → M
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (⇑((Tensor …
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    bk : Basis ι (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing.Residu …
    ⊢ Exists fun b => ∀ (i : ι), Eq (b i) (v i)
  -/
  haveI : Finite ι := Module.Finite.finite_basis bk
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    ι : Type u
    v : ι → M
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (⇑((Tensor …
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    bk : Basis ι (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing.Residu …
    this : Finite ι
    ⊢ Exists fun b => ∀ (i : ι), Eq (b i) (v i)
  -/
  letI : Fintype ι := Fintype.ofFinite ι
  letI : IsNoetherian k (k ⊗[R] (ι →₀ R)) :=
    isNoetherian_of_isNoetherianRing_of_finite k (k ⊗[R] (ι →₀ R))
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    ι : Type u
    v : ι → M
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (⇑((Tensor …
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    bk : Basis ι (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing.Residu …
    this✝¹ : Finite ι
    this✝ : Fintype ι := Fintype.ofFinite ι
    this : IsNoetherian (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing …
    ⊢ Exists fun b => ∀ (i : ι), Eq (b i) (v i)
  -/
  let i := Finsupp.linearCombination R v
  have hi : Surjective i := by
    rw [← LinearMap.range_eq_top, Finsupp.range_linearCombination]
    refine IsLocalRing.span_eq_top_of_tmul_eq_basis (R := R) (f := v) bk
      (fun _ ↦ by simp [bk])
  have : Module.Finite R (LinearMap.ker i) := by
    constructor
    exact (Submodule.fg_top _).mpr (Module.FinitePresentation.fg_ker i hi)
  -- We claim that `i` is actually a bijection,
  -- hence `v` induces an isomorphism `M ≃[R] Rᴵ` showing that `v` is a basis.
  let iequiv : (ι →₀ R) ≃ₗ[R] M := by
    refine LinearEquiv.ofBijective i ⟨?_, hi⟩
    -- By Nakayama's lemma, it suffices to show that `k ⊗ ker(i) = 0`.
    rw [← LinearMap.ker_eq_bot, ← Submodule.subsingleton_iff_eq_bot,
      ← IsLocalRing.subsingleton_tensorProduct (R := R), subsingleton_iff_forall_eq 0]
    have : Function.Surjective (i.baseChange k) := i.lTensor_surjective _ hi
    -- By construction, `k ⊗ i : kᴵ → k ⊗ M` is bijective.
    have hi' : Function.Bijective (i.baseChange k) := by
      refine ⟨?_, this⟩
      rw [← LinearMap.ker_eq_bot (M := k ⊗[R] (ι →₀ R)) (f := i.baseChange k),
        ← Submodule.finrank_eq_zero (R := k) (M := k ⊗[R] (ι →₀ R)),
        ← Nat.add_right_inj (n := Module.finrank k (LinearMap.range <| i.baseChange k)),
        LinearMap.finrank_range_add_finrank_ker (V := k ⊗[R] (ι →₀ R)),
        LinearMap.range_eq_top.mpr this, finrank_top]
      simp only [Module.finrank_tensorProduct, Module.finrank_self,
        Module.finrank_finsupp_self, one_mul, add_zero]
      rw [Module.finrank_eq_card_basis bk]
    -- On the other hand, `m ⊗ M → M` injective => `Tor₁(k, M) = 0` => `k ⊗ ker(i) → kᴵ` injective.
    intro x
    refine lTensor_injective_of_exact_of_exact_of_rTensor_injective
      (N₁ := LinearMap.ker i) (N₂ := ι →₀ R) (N₃ := M)
      (f₁ := (𝔪).subtype) (f₂ := Submodule.mkQ 𝔪)
      (g₁ := (LinearMap.ker i).subtype) (g₂ := i) (LinearMap.exact_subtype_mkQ 𝔪)
      (Submodule.mkQ_surjective _) (LinearMap.exact_subtype_ker_map i) hi H ?_ ?_
    · apply Module.Flat.lTensor_preserves_injective_linearMap
        (N := LinearMap.ker i) (N' := ι →₀ R)
        (L := (LinearMap.ker i).subtype)
      exact Subtype.val_injective
    · apply hi'.injective
      rw [LinearMap.baseChange_eq_ltensor]
      erw [← LinearMap.comp_apply (i.lTensor k), ← LinearMap.lTensor_comp]
      rw [(LinearMap.exact_subtype_ker_map i).linearMap_comp_eq_zero]
      simp only [LinearMap.lTensor_zero, LinearMap.zero_apply, map_zero]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    ι : Type u
    v : ι → M
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (⇑((Tensor …
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    bk : Basis ι (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing.Residu …
    this✝² : Finite ι
    this✝¹ : Fintype ι := Fintype.ofFinite ι
    this✝ : IsNoetherian (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRin …
    i : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R v
    hi : Function.Surjective ⇑i
    this : Module.Finite R (Subtype fun x => Membership.mem (LinearMap.ker i) x)
    iequiv : LinearEquiv (RingHom.id R) (Finsupp ι R) M := LinearEquiv.ofBijective …
    ⊢ Exists fun b => ∀ (i : ι), Eq (b i) (v i)
  -/
  use Basis.ofRepr iequiv.symm
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    ι : Type u
    v : ι → M
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (⇑((Tensor …
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    bk : Basis ι (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing.Residu …
    this✝² : Finite ι
    this✝¹ : Fintype ι := Fintype.ofFinite ι
    this✝ : IsNoetherian (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRin …
    i : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R v
    hi : Function.Surjective ⇑i
    this : Module.Finite R (Subtype fun x => Membership.mem (LinearMap.ker i) x)
    iequiv : LinearEquiv (RingHom.id R) (Finsupp ι R) M := LinearEquiv.ofBijective …
    ⊢ ∀ (i : ι), Eq ({ repr := iequiv.symm } i) (v i)
  -/
  intro j
  /-
    case h
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    ι : Type u
    v : ι → M
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (⇑((Tensor …
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    bk : Basis ι (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRing.Residu …
    this✝² : Finite ι
    this✝¹ : Fintype ι := Fintype.ofFinite ι
    this✝ : IsNoetherian (IsLocalRing.ResidueField R) (TensorProduct R (IsLocalRin …
    i : LinearMap (RingHom.id R) (Finsupp ι R) M := Finsupp.linearCombination R v
    hi : Function.Surjective ⇑i
    this : Module.Finite R (Subtype fun x => Membership.mem (LinearMap.ker i) x)
    iequiv : LinearEquiv (RingHom.id R) (Finsupp ι R) M := LinearEquiv.ofBijective …
    j : ι
    ⊢ Eq ({ repr := iequiv.symm } j) (v j)
  -/
  simp [iequiv, i]
  /-
    🎉 no goals
  -/


/--
If `M` is a finitely presented module over a local ring `(R, 𝔪)` such that `m ⊗ M → M` is
injective, then every generating family contains a basis.
-/
lemma exists_basis_of_span_of_maximalIdeal_rTensor_injective [Module.FinitePresentation R M]
    (H : Function.Injective ((𝔪).subtype.rTensor M))
    {ι : Type u} (v : ι → M) (hv : Submodule.span R (Set.range v) = ⊤) :
    ∃ (κ : Type u) (a : κ → ι) (b : Basis κ R M), ∀ i, b i = v (a i) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    ι : Type u
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    ⊢ Exists fun κ => Exists fun a => Exists fun b => ∀ (i : κ), Eq (b i) (v (a i))
  -/
  have := (map_tensorProduct_mk_eq_top (N := Submodule.span R (Set.range v))).mpr hv
  rw [← Submodule.span_image, ← Set.range_comp, eq_top_iff, ← SetLike.coe_subset_coe,
    Submodule.top_coe] at this
  have : Submodule.span k (Set.range (TensorProduct.mk R k M 1 ∘ v)) = ⊤ := by
    rw [eq_top_iff]
    exact Set.Subset.trans this (Submodule.span_subset_span _ _ _)
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    ι : Type u
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    this✝ : HasSubset.Subset Set.univ ↑(Submodule.span R (Set.range (Function.comp …
    this : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.co …
    ⊢ Exists fun κ => Exists fun a => Exists fun b => ∀ (i : κ), Eq (b i) (v (a i))
  -/
  obtain ⟨κ, a, ha, hsp, hli⟩ := exists_linearIndependent' k (TensorProduct.mk R k M 1 ∘ v)
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    ι : Type u
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    this✝ : HasSubset.Subset Set.univ ↑(Submodule.span R (Set.range (Function.comp …
    this : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.co …
    κ : Type u
    a : κ → ι
    ha : Function.Injective a
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (Function. …
    ⊢ Exists fun κ => Exists fun a => Exists fun b => ∀ (i : κ), Eq (b i) (v (a i))
  -/
  rw [this] at hsp
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    ι : Type u
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    this✝ : HasSubset.Subset Set.univ ↑(Submodule.span R (Set.range (Function.comp …
    this : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.co …
    κ : Type u
    a : κ → ι
    ha : Function.Injective a
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (Function. …
    ⊢ Exists fun κ => Exists fun a => Exists fun b => ∀ (i : κ), Eq (b i) (v (a i))
  -/
  obtain ⟨b, hb⟩ := exists_basis_of_basis_baseChange (v ∘ a) hli hsp H
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    ι : Type u
    v : ι → M
    hv : Eq (Submodule.span R (Set.range v)) Top.top
    this✝ : HasSubset.Subset Set.univ ↑(Submodule.span R (Set.range (Function.comp …
    this : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.co …
    κ : Type u
    a : κ → ι
    ha : Function.Injective a
    hsp : Eq (Submodule.span (IsLocalRing.ResidueField R) (Set.range (Function.com …
    hli : LinearIndependent (IsLocalRing.ResidueField R) (Function.comp (Function. …
    b : Basis κ R M
    hb : ∀ (i : κ), Eq (b i) (Function.comp v a i)
    ⊢ Exists fun κ => Exists fun a => Exists fun b => ∀ (i : κ), Eq (b i) (v (a i))
  -/
  use κ, a, b, hb
  /-
    🎉 no goals
  -/


lemma exists_basis_of_span_of_flat [Module.FinitePresentation R M] [Module.Flat R M]
    {ι : Type u} (v : ι → M) (hv : Submodule.span R (Set.range v) = ⊤) :
    ∃ (κ : Type u) (a : κ → ι) (b : Basis κ R M), ∀ i, b i = v (a i) :=
  exists_basis_of_span_of_maximalIdeal_rTensor_injective
    (Module.Flat.rTensor_preserves_injective_linearMap (𝔪).subtype Subtype.val_injective) v hv


/--
If `M` is a finitely presented module over a local ring `(R, 𝔪)` such that `m ⊗ M → M` is
injective, then `M` is free.
-/
theorem free_of_maximalIdeal_rTensor_injective [Module.FinitePresentation R M]
    (H : Function.Injective ((𝔪).subtype.rTensor M)) :
    Module.Free R M := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    ⊢ Module.Free R M
  -/
  obtain ⟨_, _, b, _⟩ := exists_basis_of_span_of_maximalIdeal_rTensor_injective H id (by simp)
  /-
    case intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    M : Type u_2
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : IsLocalRing R
    inst✝ : Module.FinitePresentation R M
    H : Function.Injective ⇑(LinearMap.rTensor M (Submodule.subtype (IsLocalRing.m …
    w✝¹ : Type u_2
    w✝ : w✝¹ → M
    b : Basis w✝¹ R M
    h✝ : ∀ (i : w✝¹), Eq (b i) (id (w✝ i))
    ⊢ Module.Free R M
  -/
  exact Free.of_basis b
  /-
    🎉 no goals
  -/

-- TODO: Generalise this to finite free modules.

theorem free_of_flat_of_isLocalRing [Module.FinitePresentation R P] [Module.Flat R P] :
    Module.Free R P :=
  free_of_maximalIdeal_rTensor_injective
    (Module.Flat.rTensor_preserves_injective_linearMap _ Subtype.val_injective)


@[deprecated (since := "2024-11-12")] alias free_of_flat_of_localRing := free_of_flat_of_isLocalRing


/--
If `M → N → P → 0` is a presentation of `P` over a local ring `(R, 𝔪, k)` with
`M` finite and `N` finite free, then injectivity of `k ⊗ M → k ⊗ N` implies that `P` is free.
-/
theorem free_of_lTensor_residueField_injective (hg : Surjective g) (h : Exact f g)
    [Module.Finite R M] [Module.Finite R N] [Module.Free R N]
    (hf : Function.Injective (f.lTensor k)) :
    Module.Free R P := by
  have := Module.finitePresentation_of_free_of_surjective g hg
    (by rw [h.linearMap_ker_eq, LinearMap.range_eq_map]; exact (Module.Finite.out).map f)
  /-
    R : Type u_3
    inst✝¹⁰ : CommRing R
    M : Type u_4
    N : Type u_1
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    P : Type u_2
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    inst✝³ : IsLocalRing R
    hg : Function.Surjective ⇑g
    h : Function.Exact ⇑f ⇑g
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Finite R N
    inst✝ : Module.Free R N
    hf : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) f)
    this : Module.FinitePresentation R P
    ⊢ Module.Free R P
  -/
  apply free_of_maximalIdeal_rTensor_injective
  /-
    case H
    R : Type u_3
    inst✝¹⁰ : CommRing R
    M : Type u_4
    N : Type u_1
    inst✝⁹ : AddCommGroup M
    inst✝⁸ : AddCommGroup N
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    P : Type u_2
    inst✝⁵ : AddCommGroup P
    inst✝⁴ : Module R P
    f : LinearMap (RingHom.id R) M N
    g : LinearMap (RingHom.id R) N P
    inst✝³ : IsLocalRing R
    hg : Function.Surjective ⇑g
    h : Function.Exact ⇑f ⇑g
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Finite R N
    inst✝ : Module.Free R N
    hf : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) f)
    this : Module.FinitePresentation R P
    ⊢ Function.Injective ⇑(LinearMap.rTensor P (Submodule.subtype (IsLocalRing.max …
  -/
  rw [← LinearMap.lTensor_inj_iff_rTensor_inj]
  apply lTensor_injective_of_exact_of_exact_of_rTensor_injective
    h hg (LinearMap.exact_subtype_mkQ 𝔪) (Submodule.mkQ_surjective _)
    ((LinearMap.lTensor_inj_iff_rTensor_inj _ _).mp hf)
    (Module.Flat.lTensor_preserves_injective_linearMap _ Subtype.val_injective)


/--
Given a linear map `l : M → N` over a local ring `(R, 𝔪, k)`
with `M` finite and `N` finite free,
`l` is a split injection if and only if `k ⊗ l` is a (split) injection.
-/
theorem IsLocalRing.split_injective_iff_lTensor_residueField_injective [IsLocalRing R]
    [Module.Finite R M] [Module.Finite R N] [Module.Free R N] (l : M →ₗ[R] N) :
    (∃ l', l' ∘ₗ l = LinearMap.id) ↔ Function.Injective (l.lTensor (ResidueField R)) := by
  /-
    R : Type u_1
    inst✝⁸ : CommRing R
    M : Type u_2
    N : Type u_3
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    inst✝³ : IsLocalRing R
    inst✝² : Module.Finite R M
    inst✝¹ : Module.Finite R N
    inst✝ : Module.Free R N
    l : LinearMap (RingHom.id R) M N
    ⊢ Iff (Exists fun l' => Eq (l'.comp l) LinearMap.id) (Function.Injective ⇑(Lin …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Type u_2
      N : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : IsLocalRing R
      inst✝² : Module.Finite R M
      inst✝¹ : Module.Finite R N
      inst✝ : Module.Free R N
      l : LinearMap (RingHom.id R) M N
      ⊢ (Exists fun l' => Eq (l'.comp l) LinearMap.id) → Function.Injective ⇑(Linear …
    -/
  · intro ⟨l', hl⟩
    have : l'.lTensor (ResidueField R) ∘ₗ l.lTensor (ResidueField R) = .id := by
      rw [← LinearMap.lTensor_comp, hl, LinearMap.lTensor_id]
    /-
      case mp
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Type u_2
      N : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : IsLocalRing R
      inst✝² : Module.Finite R M
      inst✝¹ : Module.Finite R N
      inst✝ : Module.Free R N
      l : LinearMap (RingHom.id R) M N
      l' : LinearMap (RingHom.id R) N M
      hl : Eq (l'.comp l) LinearMap.id
      this : Eq ((LinearMap.lTensor (IsLocalRing.ResidueField R) l').comp (LinearMap …
      ⊢ Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) l)
    -/
    exact Function.HasLeftInverse.injective ⟨_, LinearMap.congr_fun this⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Type u_2
      N : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : IsLocalRing R
      inst✝² : Module.Finite R M
      inst✝¹ : Module.Finite R N
      inst✝ : Module.Free R N
      l : LinearMap (RingHom.id R) M N
      ⊢ Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) l) → Exi …
    -/
  · intro h
    -- By `Module.free_of_lTensor_residueField_injective`, `k ⊗ l` injective => `N ⧸ l(M)` free.
    have := Module.free_of_lTensor_residueField_injective l (LinearMap.range l).mkQ
      (Submodule.mkQ_surjective _) l.exact_map_mkQ_range h
    -- Hence `l(M)` is projective because `0 → l(M) → N → N ⧸ l(M) → 0` splits.
    have : Module.Projective R (LinearMap.range l) := by
      have := (Exact.split_tfae (LinearMap.exact_subtype_mkQ (LinearMap.range l))
        Subtype.val_injective (Submodule.mkQ_surjective _)).out 0 1
      obtain ⟨l', hl'⟩ := this.mp
         (Module.projective_lifting_property _ _ (Submodule.mkQ_surjective _))
      exact Module.Projective.of_split _ _ hl'
    -- Then `0 → ker l → M → l(M) → 0` splits.
    obtain ⟨l', hl'⟩ : ∃ l', l' ∘ₗ (LinearMap.ker l).subtype = LinearMap.id := by
      have : Function.Exact (LinearMap.ker l).subtype
          (l.codRestrict (LinearMap.range l) (LinearMap.mem_range_self l)) := by
        rw [LinearMap.exact_iff, LinearMap.ker_rangeRestrict, Submodule.range_subtype]
      have := (Exact.split_tfae this
        Subtype.val_injective (fun ⟨x, y, e⟩ ↦ ⟨y, Subtype.ext e⟩)).out 0 1
      exact this.mp (Module.projective_lifting_property _ _ (fun ⟨x, y, e⟩ ↦ ⟨y, Subtype.ext e⟩))
    have : Module.Finite R (LinearMap.ker l) := by
      refine Module.Finite.of_surjective l' ?_
      exact Function.HasRightInverse.surjective ⟨_, DFunLike.congr_fun hl'⟩
    -- And tensoring with `k` preserves the injectivity of the first arrow.
    -- That is, `k ⊗ ker l → k ⊗ M` is also injective.
    have H : Function.Injective ((LinearMap.ker l).subtype.lTensor k) := by
      apply_fun (LinearMap.lTensor k) at hl'
      rw [LinearMap.lTensor_comp, LinearMap.lTensor_id] at hl'
      exact Function.HasLeftInverse.injective ⟨l'.lTensor k, DFunLike.congr_fun hl'⟩
    -- But by assumption `k ⊗ M → k ⊗ l(M)` is already injective, so `k ⊗ ker l = 0`.
    have : Subsingleton (k ⊗[R] LinearMap.ker l) := by
      refine (subsingleton_iff_forall_eq 0).mpr fun y ↦ H (h ?_)
      rw [map_zero, map_zero, ← LinearMap.comp_apply, ← LinearMap.lTensor_comp,
        l.exact_subtype_ker_map.linearMap_comp_eq_zero, LinearMap.lTensor_zero,
        LinearMap.zero_apply]
    -- By Nakayama's lemma, `l` is injective.
    have : Function.Injective l := by
      rwa [← LinearMap.ker_eq_bot, ← Submodule.subsingleton_iff_eq_bot,
        ← IsLocalRing.subsingleton_tensorProduct (R := R)]
    -- Whence `M ≃ l(M)` is projective and the result follows.
    /-
      case mpr.intro
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Type u_2
      N : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : IsLocalRing R
      inst✝² : Module.Finite R M
      inst✝¹ : Module.Finite R N
      inst✝ : Module.Free R N
      l : LinearMap (RingHom.id R) M N
      h : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) l)
      this✝³ : Module.Free R (HasQuotient.Quotient N (LinearMap.range l))
      this✝² : Module.Projective R (Subtype fun x => Membership.mem (LinearMap.range …
      l' : LinearMap (RingHom.id R) M (Subtype fun x => Membership.mem (LinearMap.ke …
      hl' : Eq (l'.comp (LinearMap.ker l).subtype) LinearMap.id
      this✝¹ : Module.Finite R (Subtype fun x => Membership.mem (LinearMap.ker l) x)
      H : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) (Linea …
      this✝ : Subsingleton (TensorProduct R (IsLocalRing.ResidueField R) (Subtype fu …
      this : Function.Injective ⇑l
      ⊢ Exists fun l' => Eq (l'.comp l) LinearMap.id
    -/
    have := (Exact.split_tfae l.exact_map_mkQ_range this (Submodule.mkQ_surjective _)).out 0 1
    /-
      case mpr.intro
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Type u_2
      N : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : IsLocalRing R
      inst✝² : Module.Finite R M
      inst✝¹ : Module.Finite R N
      inst✝ : Module.Free R N
      l : LinearMap (RingHom.id R) M N
      h : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) l)
      this✝⁴ : Module.Free R (HasQuotient.Quotient N (LinearMap.range l))
      this✝³ : Module.Projective R (Subtype fun x => Membership.mem (LinearMap.range …
      l' : LinearMap (RingHom.id R) M (Subtype fun x => Membership.mem (LinearMap.ke …
      hl' : Eq (l'.comp (LinearMap.ker l).subtype) LinearMap.id
      this✝² : Module.Finite R (Subtype fun x => Membership.mem (LinearMap.ker l) x)
      H : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) (Linea …
      this✝¹ : Subsingleton (TensorProduct R (IsLocalRing.ResidueField R) (Subtype f …
      this✝ : Function.Injective ⇑l
      this : Iff (Exists fun l_1 => Eq ((LinearMap.range l).mkQ.comp l_1) LinearMap. …
      ⊢ Exists fun l' => Eq (l'.comp l) LinearMap.id
    -/
    rw [← this]
    /-
      case mpr.intro
      R : Type u_1
      inst✝⁸ : CommRing R
      M : Type u_2
      N : Type u_3
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : IsLocalRing R
      inst✝² : Module.Finite R M
      inst✝¹ : Module.Finite R N
      inst✝ : Module.Free R N
      l : LinearMap (RingHom.id R) M N
      h : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) l)
      this✝⁴ : Module.Free R (HasQuotient.Quotient N (LinearMap.range l))
      this✝³ : Module.Projective R (Subtype fun x => Membership.mem (LinearMap.range …
      l' : LinearMap (RingHom.id R) M (Subtype fun x => Membership.mem (LinearMap.ke …
      hl' : Eq (l'.comp (LinearMap.ker l).subtype) LinearMap.id
      this✝² : Module.Finite R (Subtype fun x => Membership.mem (LinearMap.ker l) x)
      H : Function.Injective ⇑(LinearMap.lTensor (IsLocalRing.ResidueField R) (Linea …
      this✝¹ : Subsingleton (TensorProduct R (IsLocalRing.ResidueField R) (Subtype f …
      this✝ : Function.Injective ⇑l
      this : Iff (Exists fun l_1 => Eq ((LinearMap.range l).mkQ.comp l_1) LinearMap. …
      ⊢ Exists fun l_1 => Eq ((LinearMap.range l).mkQ.comp l_1) LinearMap.id
    -/
    exact Module.projective_lifting_property _ _ (Submodule.mkQ_surjective _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-09")]
alias LocalRing.split_injective_iff_lTensor_residueField_injective :=
  IsLocalRing.split_injective_iff_lTensor_residueField_injective


