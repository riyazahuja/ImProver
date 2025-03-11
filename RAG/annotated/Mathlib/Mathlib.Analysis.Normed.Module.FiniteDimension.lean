/-- A linear isometry between finite dimensional spaces of equal dimension can be upgraded
    to a linear isometry equivalence. -/
def toLinearIsometryEquiv (li : E₁ →ₗᵢ[R₁] F) (h : finrank R₁ E₁ = finrank R₁ F) :
    E₁ ≃ₗᵢ[R₁] F where
  toLinearEquiv := li.toLinearMap.linearEquivOfInjective li.injective h
  norm_map' := li.norm_map'


@[simp]
theorem coe_toLinearIsometryEquiv (li : E₁ →ₗᵢ[R₁] F) (h : finrank R₁ E₁ = finrank R₁ F) :
    (li.toLinearIsometryEquiv h : E₁ → F) = li :=
  rfl


@[simp]
theorem toLinearIsometryEquiv_apply (li : E₁ →ₗᵢ[R₁] F) (h : finrank R₁ E₁ = finrank R₁ F)
    (x : E₁) : (li.toLinearIsometryEquiv h) x = li x :=
  rfl


/-- An affine isometry between finite dimensional spaces of equal dimension can be upgraded
    to an affine isometry equivalence. -/
def toAffineIsometryEquiv [Inhabited P₁] (li : P₁ →ᵃⁱ[𝕜] P₂) (h : finrank 𝕜 V₁ = finrank 𝕜 V₂) :
    P₁ ≃ᵃⁱ[𝕜] P₂ :=
  AffineIsometryEquiv.mk' li (li.linearIsometry.toLinearIsometryEquiv h)
                                              /-
                                                𝕜 : Type u_1
                                                V₁ : Type u_2
                                                V₂ : Type u_3
                                                P₁ : Type u_4
                                                P₂ : Type u_5
                                                inst✝¹¹ : NormedField 𝕜
                                                inst✝¹⁰ : NormedAddCommGroup V₁
                                                inst✝⁹ : SeminormedAddCommGroup V₂
                                                inst✝⁸ : NormedSpace 𝕜 V₁
                                                inst✝⁷ : NormedSpace 𝕜 V₂
                                                inst✝⁶ : MetricSpace P₁
                                                inst✝⁵ : PseudoMetricSpace P₂
                                                inst✝⁴ : NormedAddTorsor V₁ P₁
                                                inst✝³ : NormedAddTorsor V₂ P₂
                                                inst✝² : FiniteDimensional 𝕜 V₁
                                                inst✝¹ : FiniteDimensional 𝕜 V₂
                                                inst✝ : Inhabited P₁
                                                li : AffineIsometry 𝕜 P₁ P₂
                                                h : Eq (Module.finrank 𝕜 V₁) (Module.finrank 𝕜 V₂)
                                                p : P₁
                                                ⊢ Eq (li p) (HVAdd.hVAdd ((li.linearIsometry.toLinearIsometryEquiv h) (VSub.vs …
                                              -/
    (Inhabited.default (α := P₁)) fun p => by simp
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
theorem coe_toAffineIsometryEquiv [Inhabited P₁] (li : P₁ →ᵃⁱ[𝕜] P₂)
    (h : finrank 𝕜 V₁ = finrank 𝕜 V₂) : (li.toAffineIsometryEquiv h : P₁ → P₂) = li :=
  rfl


@[simp]
theorem toAffineIsometryEquiv_apply [Inhabited P₁] (li : P₁ →ᵃⁱ[𝕜] P₂)
    (h : finrank 𝕜 V₁ = finrank 𝕜 V₂) (x : P₁) : (li.toAffineIsometryEquiv h) x = li x :=
  rfl


theorem AffineMap.continuous_of_finiteDimensional (f : PE →ᵃ[𝕜] PF) : Continuous f :=
  AffineMap.continuous_linear_iff.1 f.linear.continuous_of_finiteDimensional


theorem AffineEquiv.continuous_of_finiteDimensional (f : PE ≃ᵃ[𝕜] PF) : Continuous f :=
  f.toAffineMap.continuous_of_finiteDimensional


/-- Reinterpret an affine equivalence as a homeomorphism. -/
def AffineEquiv.toHomeomorphOfFiniteDimensional (f : PE ≃ᵃ[𝕜] PF) : PE ≃ₜ PF where
  toEquiv := f.toEquiv
  continuous_toFun := f.continuous_of_finiteDimensional
  continuous_invFun :=
    haveI : FiniteDimensional 𝕜 F := f.linear.finiteDimensional
    f.symm.continuous_of_finiteDimensional


@[simp]
theorem AffineEquiv.coe_toHomeomorphOfFiniteDimensional (f : PE ≃ᵃ[𝕜] PF) :
    ⇑f.toHomeomorphOfFiniteDimensional = f :=
  rfl


@[simp]
theorem AffineEquiv.coe_toHomeomorphOfFiniteDimensional_symm (f : PE ≃ᵃ[𝕜] PF) :
    ⇑f.toHomeomorphOfFiniteDimensional.symm = f.symm :=
  rfl


theorem ContinuousLinearMap.continuous_det : Continuous fun f : E →L[𝕜] E => f.det := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    ⊢ Continuous fun f => f.det
  -/
  change Continuous fun f : E →L[𝕜] E => LinearMap.det (f : E →ₗ[𝕜] E)
  -- Porting note: this could be easier with `det_cases`
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    ⊢ Continuous fun f => LinearMap.det ↑f
  -/
  by_cases h : ∃ s : Finset E, Nonempty (Basis (↥s) 𝕜 E)
    /-
      case pos
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace 𝕜
      h : Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) 𝕜 E)
      ⊢ Continuous fun f => LinearMap.det ↑f
    -/
  · rcases h with ⟨s, ⟨b⟩⟩
    /-
      case pos.intro.intro
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace 𝕜
      s : Finset E
      b : Basis (Subtype fun x => Membership.mem s x) 𝕜 E
      ⊢ Continuous fun f => LinearMap.det ↑f
    -/
    haveI : FiniteDimensional 𝕜 E := FiniteDimensional.of_fintype_basis b
    classical
    simp_rw [LinearMap.det_eq_det_toMatrix_of_finset b]
    refine Continuous.matrix_det ?_
    exact
      ((LinearMap.toMatrix b b).toLinearMap.comp
          (ContinuousLinearMap.coeLM 𝕜)).continuous_of_finiteDimensional
  · -- Porting note: was `unfold LinearMap.det`
    /-
      case neg
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace 𝕜
      h : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ Continuous fun f => LinearMap.det ↑f
    -/
    rw [LinearMap.det_def]
    /-
      case neg
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace 𝕜
      h : Not (Exists fun s => Nonempty (Basis (Subtype fun x => Membership.mem s x) …
      ⊢ Continuous fun f => (dite (Exists fun s => Nonempty (Basis (Subtype fun x => …
    -/
    simpa only [h, MonoidHom.one_apply, dif_neg, not_false_iff] using continuous_const
    /-
      🎉 no goals
    -/


/-- Any `K`-Lipschitz map from a subset `s` of a metric space `α` to a finite-dimensional real
vector space `E'` can be extended to a Lipschitz map on the whole space `α`, with a slightly worse
constant `C * K` where `C` only depends on `E'`. We record a working value for this constant `C`
as `lipschitzExtensionConstant E'`. -/
irreducible_def lipschitzExtensionConstant (E' : Type*) [NormedAddCommGroup E'] [NormedSpace ℝ E']
  [FiniteDimensional ℝ E'] : ℝ≥0 :=
  let A := (Basis.ofVectorSpace ℝ E').equivFun.toContinuousLinearEquiv
  max (‖A.symm.toContinuousLinearMap‖₊ * ‖A.toContinuousLinearMap‖₊) 1


theorem lipschitzExtensionConstant_pos (E' : Type*) [NormedAddCommGroup E'] [NormedSpace ℝ E']
    [FiniteDimensional ℝ E'] : 0 < lipschitzExtensionConstant E' := by
  /-
    E' : Type u_1
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace Real E'
    inst✝ : FiniteDimensional Real E'
    ⊢ LT.lt 0 (lipschitzExtensionConstant E')
  -/
  rw [lipschitzExtensionConstant]
  /-
    E' : Type u_1
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace Real E'
    inst✝ : FiniteDimensional Real E'
    ⊢ LT.lt 0
        (let A := (Basis.ofVectorSpace Real E').equivFun.toContinuousLinearEquiv;
        Max.max (HMul.hMul (NNNorm.nnnorm ↑A.symm) (NNNorm.nnnorm ↑A)) 1)
  -/
  exact zero_lt_one.trans_le (le_max_right _ _)
  /-
    🎉 no goals
  -/


/-- Any `K`-Lipschitz map from a subset `s` of a metric space `α` to a finite-dimensional real
vector space `E'` can be extended to a Lipschitz map on the whole space `α`, with a slightly worse
constant `lipschitzExtensionConstant E' * K`. -/
theorem LipschitzOnWith.extend_finite_dimension {α : Type*} [PseudoMetricSpace α] {E' : Type*}
    [NormedAddCommGroup E'] [NormedSpace ℝ E'] [FiniteDimensional ℝ E'] {s : Set α} {f : α → E'}
    {K : ℝ≥0} (hf : LipschitzOnWith K f s) :
    ∃ g : α → E', LipschitzWith (lipschitzExtensionConstant E' * K) g ∧ EqOn f g s := by
  /- This result is already known for spaces `ι → ℝ`. We use a continuous linear equiv between
    `E'` and such a space to transfer the result to `E'`. -/
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    E' : Type u_2
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace Real E'
    inst✝ : FiniteDimensional Real E'
    s : Set α
    f : α → E'
    K : NNReal
    hf : LipschitzOnWith K f s
    ⊢ Exists fun g => And (LipschitzWith (HMul.hMul (lipschitzExtensionConstant E' …
  -/
  let ι : Type _ := Basis.ofVectorSpaceIndex ℝ E'
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    E' : Type u_2
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace Real E'
    inst✝ : FiniteDimensional Real E'
    s : Set α
    f : α → E'
    K : NNReal
    hf : LipschitzOnWith K f s
    ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
    ⊢ Exists fun g => And (LipschitzWith (HMul.hMul (lipschitzExtensionConstant E' …
  -/
  let A := (Basis.ofVectorSpace ℝ E').equivFun.toContinuousLinearEquiv
  /-
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    E' : Type u_2
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace Real E'
    inst✝ : FiniteDimensional Real E'
    s : Set α
    f : α → E'
    K : NNReal
    hf : LipschitzOnWith K f s
    ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
    A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
    ⊢ Exists fun g => And (LipschitzWith (HMul.hMul (lipschitzExtensionConstant E' …
  -/
  have LA : LipschitzWith ‖A.toContinuousLinearMap‖₊ A := by apply A.lipschitz
  have L : LipschitzOnWith (‖A.toContinuousLinearMap‖₊ * K) (A ∘ f) s :=
    LA.comp_lipschitzOnWith hf
  obtain ⟨g, hg, gs⟩ :
    ∃ g : α → ι → ℝ, LipschitzWith (‖A.toContinuousLinearMap‖₊ * K) g ∧ EqOn (A ∘ f) g s :=
    L.extend_pi
  /-
    case intro.intro
    α : Type u_1
    inst✝³ : PseudoMetricSpace α
    E' : Type u_2
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : NormedSpace Real E'
    inst✝ : FiniteDimensional Real E'
    s : Set α
    f : α → E'
    K : NNReal
    hf : LipschitzOnWith K f s
    ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
    A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
    LA : LipschitzWith (NNNorm.nnnorm ↑A) ⇑A
    L : LipschitzOnWith (HMul.hMul (NNNorm.nnnorm ↑A) K) (Function.comp (⇑A) f) s
    g : α → ι → Real
    hg : LipschitzWith (HMul.hMul (NNNorm.nnnorm ↑A) K) g
    gs : Set.EqOn (Function.comp (⇑A) f) g s
    ⊢ Exists fun g => And (LipschitzWith (HMul.hMul (lipschitzExtensionConstant E' …
  -/
  refine ⟨A.symm ∘ g, ?_, ?_⟩
  · have LAsymm : LipschitzWith ‖A.symm.toContinuousLinearMap‖₊ A.symm := by
      apply A.symm.lipschitz
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      E' : Type u_2
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace Real E'
      inst✝ : FiniteDimensional Real E'
      s : Set α
      f : α → E'
      K : NNReal
      hf : LipschitzOnWith K f s
      ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
      A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
      LA : LipschitzWith (NNNorm.nnnorm ↑A) ⇑A
      L : LipschitzOnWith (HMul.hMul (NNNorm.nnnorm ↑A) K) (Function.comp (⇑A) f) s
      g : α → ι → Real
      hg : LipschitzWith (HMul.hMul (NNNorm.nnnorm ↑A) K) g
      gs : Set.EqOn (Function.comp (⇑A) f) g s
      LAsymm : LipschitzWith (NNNorm.nnnorm ↑A.symm) ⇑A.symm
      ⊢ LipschitzWith (HMul.hMul (lipschitzExtensionConstant E') K) (Function.comp ( …
    -/
    apply (LAsymm.comp hg).weaken
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      E' : Type u_2
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace Real E'
      inst✝ : FiniteDimensional Real E'
      s : Set α
      f : α → E'
      K : NNReal
      hf : LipschitzOnWith K f s
      ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
      A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
      LA : LipschitzWith (NNNorm.nnnorm ↑A) ⇑A
      L : LipschitzOnWith (HMul.hMul (NNNorm.nnnorm ↑A) K) (Function.comp (⇑A) f) s
      g : α → ι → Real
      hg : LipschitzWith (HMul.hMul (NNNorm.nnnorm ↑A) K) g
      gs : Set.EqOn (Function.comp (⇑A) f) g s
      LAsymm : LipschitzWith (NNNorm.nnnorm ↑A.symm) ⇑A.symm
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm ↑A.symm) (HMul.hMul (NNNorm.nnnorm ↑A) K)) ( …
    -/
    rw [lipschitzExtensionConstant, ← mul_assoc]
    /-
      case intro.intro.refine_1
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      E' : Type u_2
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace Real E'
      inst✝ : FiniteDimensional Real E'
      s : Set α
      f : α → E'
      K : NNReal
      hf : LipschitzOnWith K f s
      ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
      A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
      LA : LipschitzWith (NNNorm.nnnorm ↑A) ⇑A
      L : LipschitzOnWith (HMul.hMul (NNNorm.nnnorm ↑A) K) (Function.comp (⇑A) f) s
      g : α → ι → Real
      hg : LipschitzWith (HMul.hMul (NNNorm.nnnorm ↑A) K) g
      gs : Set.EqOn (Function.comp (⇑A) f) g s
      LAsymm : LipschitzWith (NNNorm.nnnorm ↑A.symm) ⇑A.symm
      ⊢ LE.le (HMul.hMul (HMul.hMul (NNNorm.nnnorm ↑A.symm) (NNNorm.nnnorm ↑A)) K)
          (HMul.hMul
            (let A := (Basis.ofVectorSpace Real E').equivFun.toContinuousLinearEquiv;
            Max.max (HMul.hMul (NNNorm.nnnorm ↑A.symm) (NNNorm.nnnorm ↑A)) 1)
            K)
    -/
    exact mul_le_mul' (le_max_left _ _) le_rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      E' : Type u_2
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace Real E'
      inst✝ : FiniteDimensional Real E'
      s : Set α
      f : α → E'
      K : NNReal
      hf : LipschitzOnWith K f s
      ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
      A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
      LA : LipschitzWith (NNNorm.nnnorm ↑A) ⇑A
      L : LipschitzOnWith (HMul.hMul (NNNorm.nnnorm ↑A) K) (Function.comp (⇑A) f) s
      g : α → ι → Real
      hg : LipschitzWith (HMul.hMul (NNNorm.nnnorm ↑A) K) g
      gs : Set.EqOn (Function.comp (⇑A) f) g s
      ⊢ Set.EqOn f (Function.comp (⇑A.symm) g) s
    -/
  · intro x hx
    /-
      case intro.intro.refine_2
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      E' : Type u_2
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace Real E'
      inst✝ : FiniteDimensional Real E'
      s : Set α
      f : α → E'
      K : NNReal
      hf : LipschitzOnWith K f s
      ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
      A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
      LA : LipschitzWith (NNNorm.nnnorm ↑A) ⇑A
      L : LipschitzOnWith (HMul.hMul (NNNorm.nnnorm ↑A) K) (Function.comp (⇑A) f) s
      g : α → ι → Real
      hg : LipschitzWith (HMul.hMul (NNNorm.nnnorm ↑A) K) g
      gs : Set.EqOn (Function.comp (⇑A) f) g s
      x : α
      hx : Membership.mem s x
      ⊢ Eq (f x) (Function.comp (⇑A.symm) g x)
    -/
    have : A (f x) = g x := gs hx
    /-
      case intro.intro.refine_2
      α : Type u_1
      inst✝³ : PseudoMetricSpace α
      E' : Type u_2
      inst✝² : NormedAddCommGroup E'
      inst✝¹ : NormedSpace Real E'
      inst✝ : FiniteDimensional Real E'
      s : Set α
      f : α → E'
      K : NNReal
      hf : LipschitzOnWith K f s
      ι : Type u_2 := ↑(Basis.ofVectorSpaceIndex Real E')
      A : ContinuousLinearEquiv (RingHom.id Real) E' (↑(Basis.ofVectorSpaceIndex Rea …
      LA : LipschitzWith (NNNorm.nnnorm ↑A) ⇑A
      L : LipschitzOnWith (HMul.hMul (NNNorm.nnnorm ↑A) K) (Function.comp (⇑A) f) s
      g : α → ι → Real
      hg : LipschitzWith (HMul.hMul (NNNorm.nnnorm ↑A) K) g
      gs : Set.EqOn (Function.comp (⇑A) f) g s
      x : α
      hx : Membership.mem s x
      this : Eq (A (f x)) (g x)
      ⊢ Eq (f x) (Function.comp (⇑A.symm) g x)
    -/
    simp only [(· ∘ ·), ← this, A.symm_apply_apply]
    /-
      🎉 no goals
    -/


theorem LinearMap.exists_antilipschitzWith [FiniteDimensional 𝕜 E] (f : E →ₗ[𝕜] F)
    (hf : LinearMap.ker f = ⊥) : ∃ K > 0, AntilipschitzWith K f := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F
    hf : Eq (LinearMap.ker f) Bot.bot
    ⊢ Exists fun K => And (GT.gt K 0) (AntilipschitzWith K ⇑f)
  -/
  cases subsingleton_or_nontrivial E
    /-
      case inl
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      hf : Eq (LinearMap.ker f) Bot.bot
      h✝ : Subsingleton E
      ⊢ Exists fun K => And (GT.gt K 0) (AntilipschitzWith K ⇑f)
    -/
  · exact ⟨1, zero_lt_one, AntilipschitzWith.of_subsingleton⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      hf : Eq (LinearMap.ker f) Bot.bot
      h✝ : Nontrivial E
      ⊢ Exists fun K => And (GT.gt K 0) (AntilipschitzWith K ⇑f)
    -/
  · rw [LinearMap.ker_eq_bot] at hf
    /-
      case inr
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      hf : Function.Injective ⇑f
      h✝ : Nontrivial E
      ⊢ Exists fun K => And (GT.gt K 0) (AntilipschitzWith K ⇑f)
    -/
    let e : E ≃L[𝕜] LinearMap.range f := (LinearEquiv.ofInjective f hf).toContinuousLinearEquiv
    /-
      case inr
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      hf : Function.Injective ⇑f
      h✝ : Nontrivial E
      e : ContinuousLinearEquiv (RingHom.id 𝕜) E (Subtype fun x => Membership.mem (L …
      ⊢ Exists fun K => And (GT.gt K 0) (AntilipschitzWith K ⇑f)
    -/
    exact ⟨_, e.nnnorm_symm_pos, e.antilipschitz⟩
    /-
      🎉 no goals
    -/


open Function in
/-- A `LinearMap` on a finite-dimensional space over a complete field
  is injective iff it is anti-Lipschitz. -/
theorem LinearMap.injective_iff_antilipschitz [FiniteDimensional 𝕜 E] (f : E →ₗ[𝕜] F) :
    Injective f ↔ ∃ K > 0, AntilipschitzWith K f := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    f : LinearMap (RingHom.id 𝕜) E F
    ⊢ Iff (Function.Injective ⇑f) (Exists fun K => And (GT.gt K 0) (AntilipschitzW …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      ⊢ Function.Injective ⇑f → Exists fun K => And (GT.gt K 0) (AntilipschitzWith K …
    -/
  · rw [← LinearMap.ker_eq_bot]
    /-
      case mp
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      ⊢ Eq (LinearMap.ker f) Bot.bot → Exists fun K => And (GT.gt K 0) (Antilipschit …
    -/
    exact f.exists_antilipschitzWith
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      ⊢ (Exists fun K => And (GT.gt K 0) (AntilipschitzWith K ⇑f)) → Function.Inject …
    -/
  · rintro ⟨K, -, H⟩
    /-
      case mpr.intro.intro
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      inst✝ : FiniteDimensional 𝕜 E
      f : LinearMap (RingHom.id 𝕜) E F
      K : NNReal
      H : AntilipschitzWith K ⇑f
      ⊢ Function.Injective ⇑f
    -/
    exact H.injective
    /-
      🎉 no goals
    -/


open Function in
/-- The set of injective continuous linear maps `E → F` is open,
  if `E` is finite-dimensional over a complete field. -/
theorem ContinuousLinearMap.isOpen_injective [FiniteDimensional 𝕜 E] :
    IsOpen { L : E →L[𝕜] F | Injective L } := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ IsOpen (setOf fun L => Function.Injective ⇑L)
  -/
  rw [isOpen_iff_eventually]
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ ∀ (x : ContinuousLinearMap (RingHom.id 𝕜) E F), Membership.mem (setOf fun L  …
  -/
  rintro φ₀ hφ₀
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    φ₀ : ContinuousLinearMap (RingHom.id 𝕜) E F
    hφ₀ : Membership.mem (setOf fun L => Function.Injective ⇑L) φ₀
    ⊢ Filter.Eventually (fun y => Membership.mem (setOf fun L => Function.Injectiv …
  -/
  rcases φ₀.injective_iff_antilipschitz.mp hφ₀ with ⟨K, K_pos, H⟩
  /-
    case intro.intro
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    φ₀ : ContinuousLinearMap (RingHom.id 𝕜) E F
    hφ₀ : Membership.mem (setOf fun L => Function.Injective ⇑L) φ₀
    K : NNReal
    K_pos : GT.gt K 0
    H : AntilipschitzWith K ⇑↑φ₀
    ⊢ Filter.Eventually (fun y => Membership.mem (setOf fun L => Function.Injectiv …
  -/
  have : ∀ᶠ φ in 𝓝 φ₀, ‖φ - φ₀‖₊ < K⁻¹ := eventually_nnnorm_sub_lt _ <| inv_pos_of_pos K_pos
  /-
    case intro.intro
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    φ₀ : ContinuousLinearMap (RingHom.id 𝕜) E F
    hφ₀ : Membership.mem (setOf fun L => Function.Injective ⇑L) φ₀
    K : NNReal
    K_pos : GT.gt K 0
    H : AntilipschitzWith K ⇑↑φ₀
    this : Filter.Eventually (fun φ => LT.lt (NNNorm.nnnorm (HSub.hSub φ φ₀)) (Inv …
    ⊢ Filter.Eventually (fun y => Membership.mem (setOf fun L => Function.Injectiv …
  -/
  filter_upwards [this] with φ hφ
  /-
    case h
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    φ₀ : ContinuousLinearMap (RingHom.id 𝕜) E F
    hφ₀ : Membership.mem (setOf fun L => Function.Injective ⇑L) φ₀
    K : NNReal
    K_pos : GT.gt K 0
    H : AntilipschitzWith K ⇑↑φ₀
    this : Filter.Eventually (fun φ => LT.lt (NNNorm.nnnorm (HSub.hSub φ φ₀)) (Inv …
    φ : ContinuousLinearMap (RingHom.id 𝕜) E F
    hφ : LT.lt (NNNorm.nnnorm (HSub.hSub φ φ₀)) (Inv.inv K)
    ⊢ Function.Injective ⇑φ
  -/
  apply φ.injective_iff_antilipschitz.mpr
  exact ⟨(K⁻¹ - ‖φ - φ₀‖₊)⁻¹, inv_pos_of_pos (tsub_pos_of_lt hφ),
    H.add_sub_lipschitzWith (φ - φ₀).lipschitz hφ⟩


protected theorem LinearIndependent.eventually {ι} [Finite ι] {f : ι → E}
    (hf : LinearIndependent 𝕜 f) : ∀ᶠ g in 𝓝 f, LinearIndependent 𝕜 g := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : CompleteSpace 𝕜
    ι : Type u_1
    inst✝ : Finite ι
    f : ι → E
    hf : LinearIndependent 𝕜 f
    ⊢ Filter.Eventually (fun g => LinearIndependent 𝕜 g) (nhds f)
  -/
  cases nonempty_fintype ι
  classical
  simp only [Fintype.linearIndependent_iff'] at hf ⊢
  rcases LinearMap.exists_antilipschitzWith _ hf with ⟨K, K0, hK⟩
  have : Tendsto (fun g : ι → E => ∑ i, ‖g i - f i‖) (𝓝 f) (𝓝 <| ∑ i, ‖f i - f i‖) :=
    tendsto_finset_sum _ fun i _ =>
      Tendsto.norm <| ((continuous_apply i).tendsto _).sub tendsto_const_nhds
  simp only [sub_self, norm_zero, Finset.sum_const_zero] at this
  refine (this.eventually (gt_mem_nhds <| inv_pos.2 K0)).mono fun g hg => ?_
  replace hg : ∑ i, ‖g i - f i‖₊ < K⁻¹ := by
    rw [← NNReal.coe_lt_coe]
    push_cast
    exact hg
  rw [LinearMap.ker_eq_bot]
  refine (hK.add_sub_lipschitzWith (LipschitzWith.of_dist_le_mul fun v u => ?_) hg).injective
  simp only [dist_eq_norm, LinearMap.lsum_apply, Pi.sub_apply, LinearMap.sum_apply,
    LinearMap.comp_apply, LinearMap.proj_apply, LinearMap.smulRight_apply, LinearMap.id_apply, ←
    Finset.sum_sub_distrib, ← smul_sub, ← sub_smul, NNReal.coe_sum, coe_nnnorm, Finset.sum_mul]
  refine norm_sum_le_of_le _ fun i _ => ?_
  rw [norm_smul, mul_comm]
  gcongr
  exact norm_le_pi_norm (v - u) i


theorem isOpen_setOf_linearIndependent {ι : Type*} [Finite ι] :
    IsOpen { f : ι → E | LinearIndependent 𝕜 f } :=
  isOpen_iff_mem_nhds.2 fun _ => LinearIndependent.eventually


theorem isOpen_setOf_nat_le_rank (n : ℕ) :
    IsOpen { f : E →L[𝕜] F | ↑n ≤ (f : E →ₗ[𝕜] F).rank } := by
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type w
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace 𝕜
    n : Nat
    ⊢ IsOpen (setOf fun f => LE.le (↑n) (↑f).rank)
  -/
  simp only [LinearMap.le_rank_iff_exists_linearIndependent_finset, setOf_exists, ← exists_prop]
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type w
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace 𝕜
    n : Nat
    ⊢ IsOpen (Set.iUnion fun i => Set.iUnion fun i_1 => setOf fun x => LinearIndep …
  -/
  refine isOpen_biUnion fun t _ => ?_
  have : Continuous fun f : E →L[𝕜] F => fun x : (t : Set E) => f x :=
    continuous_pi fun x => (ContinuousLinearMap.apply 𝕜 F (x : E)).continuous
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type w
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace 𝕜
    n : Nat
    t : Finset E
    x✝ : Membership.mem (fun i => Eq i.card n) t
    this : Continuous fun f x => f ↑x
    ⊢ IsOpen (setOf fun x => LinearIndependent 𝕜 fun x_1 => ↑x ↑x_1)
  -/
  exact isOpen_setOf_linearIndependent.preimage this
  /-
    🎉 no goals
  -/


theorem Basis.opNNNorm_le {ι : Type*} [Fintype ι] (v : Basis ι 𝕜 E) {u : E →L[𝕜] F} (M : ℝ≥0)
    (hu : ∀ i, ‖u (v i)‖₊ ≤ M) : ‖u‖₊ ≤ Fintype.card ι • ‖v.equivFunL.toContinuousLinearMap‖₊ * M :=
  u.opNNNorm_le_bound _ fun e => by
    /-
      𝕜 : Type u
      inst✝⁶ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      F : Type w
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : CompleteSpace 𝕜
      ι : Type u_1
      inst✝ : Fintype ι
      v : Basis ι 𝕜 E
      u : ContinuousLinearMap (RingHom.id 𝕜) E F
      M : NNReal
      hu : ∀ (i : ι), LE.le (NNNorm.nnnorm (u (v i))) M
      e : E
      ⊢ LE.le (NNNorm.nnnorm (u e)) (HMul.hMul (HMul.hMul (HSMul.hSMul (Fintype.card …
    -/
    set φ := v.equivFunL.toContinuousLinearMap
    calc
      ‖u e‖₊ = ‖u (∑ i, v.equivFun e i • v i)‖₊ := by rw [v.sum_equivFun]
      _ = ‖∑ i, v.equivFun e i • (u <| v i)‖₊ := by simp [map_sum, LinearMap.map_smul]
      _ ≤ ∑ i, ‖v.equivFun e i • (u <| v i)‖₊ := nnnorm_sum_le _ _
      _ = ∑ i, ‖v.equivFun e i‖₊ * ‖u (v i)‖₊ := by simp only [nnnorm_smul]
      _ ≤ ∑ i, ‖v.equivFun e i‖₊ * M := by gcongr; apply hu
      _ = (∑ i, ‖v.equivFun e i‖₊) * M := by rw [Finset.sum_mul]
      _ ≤ Fintype.card ι • (‖φ‖₊ * ‖e‖₊) * M := by
        gcongr
        calc
          ∑ i, ‖v.equivFun e i‖₊ ≤ Fintype.card ι • ‖φ e‖₊ := Pi.sum_nnnorm_apply_le_nnnorm _
          _ ≤ Fintype.card ι • (‖φ‖₊ * ‖e‖₊) := nsmul_le_nsmul_right (φ.le_opNNNorm e) _
      _ = Fintype.card ι • ‖φ‖₊ * M * ‖e‖₊ := by simp only [smul_mul_assoc, mul_right_comm]


@[deprecated (since := "2024-02-02")] alias Basis.op_nnnorm_le := Basis.opNNNorm_le


theorem Basis.opNorm_le {ι : Type*} [Fintype ι] (v : Basis ι 𝕜 E) {u : E →L[𝕜] F} {M : ℝ}
    (hM : 0 ≤ M) (hu : ∀ i, ‖u (v i)‖ ≤ M) :
    ‖u‖ ≤ Fintype.card ι • ‖v.equivFunL.toContinuousLinearMap‖ * M := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    ι : Type u_1
    inst✝ : Fintype ι
    v : Basis ι 𝕜 E
    u : ContinuousLinearMap (RingHom.id 𝕜) E F
    M : Real
    hM : LE.le 0 M
    hu : ∀ (i : ι), LE.le (Norm.norm (u (v i))) M
    ⊢ LE.le (Norm.norm u) (HMul.hMul (HSMul.hSMul (Fintype.card ι) (Norm.norm ↑v.e …
  -/
  simpa using NNReal.coe_le_coe.mpr (v.opNNNorm_le ⟨M, hM⟩ hu)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias Basis.op_norm_le := Basis.opNorm_le


/-- A weaker version of `Basis.opNNNorm_le` that abstracts away the value of `C`. -/
theorem Basis.exists_opNNNorm_le {ι : Type*} [Finite ι] (v : Basis ι 𝕜 E) :
    ∃ C > (0 : ℝ≥0), ∀ {u : E →L[𝕜] F} (M : ℝ≥0), (∀ i, ‖u (v i)‖₊ ≤ M) → ‖u‖₊ ≤ C * M := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    ι : Type u_1
    inst✝ : Finite ι
    v : Basis ι 𝕜 E
    ⊢ Exists fun C => And (GT.gt C 0) (∀ {u : ContinuousLinearMap (RingHom.id 𝕜) E …
  -/
  cases nonempty_fintype ι
  exact
    ⟨max (Fintype.card ι • ‖v.equivFunL.toContinuousLinearMap‖₊) 1,
      zero_lt_one.trans_le (le_max_right _ _), fun {u} M hu =>
      (v.opNNNorm_le M hu).trans <| mul_le_mul_of_nonneg_right (le_max_left _ _) (zero_le M)⟩


@[deprecated (since := "2024-02-02")] alias Basis.exists_op_nnnorm_le := Basis.exists_opNNNorm_le


/-- A weaker version of `Basis.opNorm_le` that abstracts away the value of `C`. -/
theorem Basis.exists_opNorm_le {ι : Type*} [Finite ι] (v : Basis ι 𝕜 E) :
    ∃ C > (0 : ℝ), ∀ {u : E →L[𝕜] F} {M : ℝ}, 0 ≤ M → (∀ i, ‖u (v i)‖ ≤ M) → ‖u‖ ≤ C * M := by
  /-
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    ι : Type u_1
    inst✝ : Finite ι
    v : Basis ι 𝕜 E
    ⊢ Exists fun C => And (GT.gt C 0) (∀ {u : ContinuousLinearMap (RingHom.id 𝕜) E …
  -/
  obtain ⟨C, hC, h⟩ := v.exists_opNNNorm_le (F := F)
  -- Porting note: used `Subtype.forall'` below
  /-
    case intro.intro
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    ι : Type u_1
    inst✝ : Finite ι
    v : Basis ι 𝕜 E
    C : NNReal
    hC : GT.gt C 0
    h : ∀ {u : ContinuousLinearMap (RingHom.id 𝕜) E F} (M : NNReal), (∀ (i : ι), L …
    ⊢ Exists fun C => And (GT.gt C 0) (∀ {u : ContinuousLinearMap (RingHom.id 𝕜) E …
  -/
  refine ⟨C, hC, ?_⟩
  /-
    case intro.intro
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    ι : Type u_1
    inst✝ : Finite ι
    v : Basis ι 𝕜 E
    C : NNReal
    hC : GT.gt C 0
    h : ∀ {u : ContinuousLinearMap (RingHom.id 𝕜) E F} (M : NNReal), (∀ (i : ι), L …
    ⊢ ∀ {u : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ (i …
  -/
  intro u M hM H
  /-
    case intro.intro
    𝕜 : Type u
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type w
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace 𝕜
    ι : Type u_1
    inst✝ : Finite ι
    v : Basis ι 𝕜 E
    C : NNReal
    hC : GT.gt C 0
    h : ∀ {u : ContinuousLinearMap (RingHom.id 𝕜) E F} (M : NNReal), (∀ (i : ι), L …
    u : ContinuousLinearMap (RingHom.id 𝕜) E F
    M : Real
    hM : LE.le 0 M
    H : ∀ (i : ι), LE.le (Norm.norm (u (v i))) M
    ⊢ LE.le (Norm.norm u) (HMul.hMul (↑C) M)
  -/
  simpa using h ⟨M, hM⟩ H
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias Basis.exists_op_norm_le := Basis.exists_opNorm_le


instance [FiniteDimensional 𝕜 E] [SecondCountableTopology F] :
    SecondCountableTopology (E →L[𝕜] F) := by
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    ⊢ SecondCountableTopology (ContinuousLinearMap (RingHom.id 𝕜) E F)
  -/
  set d := Module.finrank 𝕜 E
  suffices
    ∀ ε > (0 : ℝ), ∃ n : (E →L[𝕜] F) → Fin d → ℕ, ∀ f g : E →L[𝕜] F, n f = n g → dist f g ≤ ε from
    Metric.secondCountable_of_countable_discretization fun ε ε_pos =>
      ⟨Fin d → ℕ, by infer_instance, this ε ε_pos⟩
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun n => ∀ (f g : ContinuousLinearMap (Ring …
  -/
  intro ε ε_pos
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  obtain ⟨u : ℕ → F, hu : DenseRange u⟩ := exists_dense_seq F
  /-
    case intro
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  let v := Module.finBasis 𝕜 E
  obtain
    ⟨C : ℝ, C_pos : 0 < C, hC :
      ∀ {φ : E →L[𝕜] F} {M : ℝ}, 0 ≤ M → (∀ i, ‖φ (v i)‖ ≤ M) → ‖φ‖ ≤ C * M⟩ :=
    v.exists_opNorm_le (E := E) (F := F)
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    v : Basis (Fin (Module.finrank 𝕜 E)) 𝕜 E := Module.finBasis 𝕜 E
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ {φ : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ …
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  have h_2C : 0 < 2 * C := mul_pos zero_lt_two C_pos
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    v : Basis (Fin (Module.finrank 𝕜 E)) 𝕜 E := Module.finBasis 𝕜 E
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ {φ : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ …
    h_2C : LT.lt 0 (HMul.hMul 2 C)
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  have hε2C : 0 < ε / (2 * C) := div_pos ε_pos h_2C
  have : ∀ φ : E →L[𝕜] F, ∃ n : Fin d → ℕ, ‖φ - (v.constrL <| u ∘ n)‖ ≤ ε / 2 := by
    intro φ
    have : ∀ i, ∃ n, ‖φ (v i) - u n‖ ≤ ε / (2 * C) := by
      simp only [norm_sub_rev]
      intro i
      have : φ (v i) ∈ closure (range u) := hu _
      obtain ⟨n, hn⟩ : ∃ n, ‖u n - φ (v i)‖ < ε / (2 * C) := by
        rw [mem_closure_iff_nhds_basis Metric.nhds_basis_ball] at this
        specialize this (ε / (2 * C)) hε2C
        simpa [dist_eq_norm]
      exact ⟨n, le_of_lt hn⟩
    choose n hn using this
    use n
    replace hn : ∀ i : Fin d, ‖(φ - (v.constrL <| u ∘ n)) (v i)‖ ≤ ε / (2 * C) := by simp [hn]
    have : C * (ε / (2 * C)) = ε / 2 := by
      rw [eq_div_iff (two_ne_zero : (2 : ℝ) ≠ 0), mul_comm, ← mul_assoc,
        mul_div_cancel₀ _ (ne_of_gt h_2C)]
    specialize hC (le_of_lt hε2C) hn
    rwa [this] at hC
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    v : Basis (Fin (Module.finrank 𝕜 E)) 𝕜 E := Module.finBasis 𝕜 E
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ {φ : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ …
    h_2C : LT.lt 0 (HMul.hMul 2 C)
    hε2C : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 C))
    this : ∀ (φ : ContinuousLinearMap (RingHom.id 𝕜) E F), Exists fun n => LE.le ( …
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  choose n hn using this
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    v : Basis (Fin (Module.finrank 𝕜 E)) 𝕜 E := Module.finBasis 𝕜 E
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ {φ : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ …
    h_2C : LT.lt 0 (HMul.hMul 2 C)
    hε2C : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 C))
    n : ContinuousLinearMap (RingHom.id 𝕜) E F → Fin d → Nat
    hn : ∀ (φ : ContinuousLinearMap (RingHom.id 𝕜) E F), LE.le (Norm.norm (HSub.hS …
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  set Φ := fun φ : E →L[𝕜] F => v.constrL <| u ∘ n φ
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    v : Basis (Fin (Module.finrank 𝕜 E)) 𝕜 E := Module.finBasis 𝕜 E
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ {φ : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ …
    h_2C : LT.lt 0 (HMul.hMul 2 C)
    hε2C : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 C))
    n : ContinuousLinearMap (RingHom.id 𝕜) E F → Fin d → Nat
    hn : ∀ (φ : ContinuousLinearMap (RingHom.id 𝕜) E F), LE.le (Norm.norm (HSub.hS …
    Φ : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  change ∀ z, dist z (Φ z) ≤ ε / 2 at hn
  /-
    case intro.intro.intro
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    v : Basis (Fin (Module.finrank 𝕜 E)) 𝕜 E := Module.finBasis 𝕜 E
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ {φ : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ …
    h_2C : LT.lt 0 (HMul.hMul 2 C)
    hε2C : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 C))
    n : ContinuousLinearMap (RingHom.id 𝕜) E F → Fin d → Nat
    Φ : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
    hn : ∀ (z : ContinuousLinearMap (RingHom.id 𝕜) E F), LE.le (Dist.dist z (Φ z)) …
    ⊢ Exists fun n => ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) ( …
  -/
  use n
  /-
    case h
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    inst✝¹ : FiniteDimensional 𝕜 E
    inst✝ : SecondCountableTopology F
    d : Nat := Module.finrank 𝕜 E
    ε : Real
    ε_pos : GT.gt ε 0
    u : Nat → F
    hu : DenseRange u
    v : Basis (Fin (Module.finrank 𝕜 E)) 𝕜 E := Module.finBasis 𝕜 E
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ {φ : ContinuousLinearMap (RingHom.id 𝕜) E F} {M : Real}, LE.le 0 M → (∀ …
    h_2C : LT.lt 0 (HMul.hMul 2 C)
    hε2C : LT.lt 0 (HDiv.hDiv ε (HMul.hMul 2 C))
    n : ContinuousLinearMap (RingHom.id 𝕜) E F → Fin d → Nat
    Φ : ContinuousLinearMap (RingHom.id 𝕜) E F → ContinuousLinearMap (RingHom.id 𝕜 …
    hn : ∀ (z : ContinuousLinearMap (RingHom.id 𝕜) E F), LE.le (Dist.dist z (Φ z)) …
    ⊢ ∀ (f g : ContinuousLinearMap (RingHom.id 𝕜) E F), Eq (n f) (n g) → LE.le (Di …
  -/
  intro x y hxy
  calc
    dist x y ≤ dist x (Φ x) + dist (Φ x) y := dist_triangle _ _ _
    _ = dist x (Φ x) + dist y (Φ y) := by simp [Φ, hxy, dist_comm]
    _ ≤ ε := by linarith [hn x, hn y]


theorem AffineSubspace.closed_of_finiteDimensional {P : Type*} [MetricSpace P]
    [NormedAddTorsor E P] (s : AffineSubspace 𝕜 P) [FiniteDimensional 𝕜 s.direction] :
    IsClosed (s : Set P) :=
  s.isClosed_direction_iff.mp s.direction.closed_of_finiteDimensional


/-- In an infinite dimensional space, given a finite number of points, one may find a point
with norm at most `R` which is at distance at least `1` of all these points. -/
theorem exists_norm_le_le_norm_sub_of_finset {c : 𝕜} (hc : 1 < ‖c‖) {R : ℝ} (hR : ‖c‖ < R)
    (h : ¬FiniteDimensional 𝕜 E) (s : Finset E) : ∃ x : E, ‖x‖ ≤ R ∧ ∀ y ∈ s, 1 ≤ ‖y - x‖ := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    h : Not (FiniteDimensional 𝕜 E)
    s : Finset E
    ⊢ Exists fun x => And (LE.le (Norm.norm x) R) (∀ (y : E), Membership.mem s y → …
  -/
  let F := Submodule.span 𝕜 (s : Set E)
  haveI : FiniteDimensional 𝕜 F :=
    Module.finite_def.2
      ((Submodule.fg_top _).2 (Submodule.fg_def.2 ⟨s, Finset.finite_toSet _, rfl⟩))
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    h : Not (FiniteDimensional 𝕜 E)
    s : Finset E
    F : Submodule 𝕜 E := Submodule.span 𝕜 ↑s
    this : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem F x)
    ⊢ Exists fun x => And (LE.le (Norm.norm x) R) (∀ (y : E), Membership.mem s y → …
  -/
  have Fclosed : IsClosed (F : Set E) := Submodule.closed_of_finiteDimensional _
  have : ∃ x, x ∉ F := by
    contrapose! h
    have : (⊤ : Submodule 𝕜 E) = F := by
      ext x
      simp [h]
    have : FiniteDimensional 𝕜 (⊤ : Submodule 𝕜 E) := by rwa [this]
    exact Module.finite_def.2 ((Submodule.fg_top _).1 (Module.finite_def.1 this))
  obtain ⟨x, xR, hx⟩ : ∃ x : E, ‖x‖ ≤ R ∧ ∀ y : E, y ∈ F → 1 ≤ ‖x - y‖ :=
    riesz_lemma_of_norm_lt hc hR Fclosed this
  have hx' : ∀ y : E, y ∈ F → 1 ≤ ‖y - x‖ := by
    intro y hy
    rw [← norm_neg]
    simpa using hx y hy
  /-
    case intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    h : Not (FiniteDimensional 𝕜 E)
    s : Finset E
    F : Submodule 𝕜 E := Submodule.span 𝕜 ↑s
    this✝ : FiniteDimensional 𝕜 (Subtype fun x => Membership.mem F x)
    Fclosed : IsClosed ↑F
    this : Exists fun x => Not (Membership.mem F x)
    x : E
    xR : LE.le (Norm.norm x) R
    hx : ∀ (y : E), Membership.mem F y → LE.le 1 (Norm.norm (HSub.hSub x y))
    hx' : ∀ (y : E), Membership.mem F y → LE.le 1 (Norm.norm (HSub.hSub y x))
    ⊢ Exists fun x => And (LE.le (Norm.norm x) R) (∀ (y : E), Membership.mem s y → …
  -/
  exact ⟨x, xR, fun y hy => hx' _ (Submodule.subset_span hy)⟩
  /-
    🎉 no goals
  -/


/-- In an infinite-dimensional normed space, there exists a sequence of points which are all
bounded by `R` and at distance at least `1`. For a version not assuming `c` and `R`, see
`exists_seq_norm_le_one_le_norm_sub`. -/
theorem exists_seq_norm_le_one_le_norm_sub' {c : 𝕜} (hc : 1 < ‖c‖) {R : ℝ} (hR : ‖c‖ < R)
    (h : ¬FiniteDimensional 𝕜 E) :
    ∃ f : ℕ → E, (∀ n, ‖f n‖ ≤ R) ∧ Pairwise fun m n => 1 ≤ ‖f m - f n‖ := by
  have : IsSymm E fun x y : E => 1 ≤ ‖x - y‖ := by
    constructor
    intro x y hxy
    rw [← norm_neg]
    simpa
  apply
    exists_seq_of_forall_finset_exists' (fun x : E => ‖x‖ ≤ R) fun (x : E) (y : E) => 1 ≤ ‖x - y‖
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    h : Not (FiniteDimensional 𝕜 E)
    this : IsSymm E fun x y => LE.le 1 (Norm.norm (HSub.hSub x y))
    ⊢ ∀ (s : Finset E), (∀ (x : E), Membership.mem s x → LE.le (Norm.norm x) R) →  …
  -/
  rintro s -
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    R : Real
    hR : LT.lt (Norm.norm c) R
    h : Not (FiniteDimensional 𝕜 E)
    this : IsSymm E fun x y => LE.le 1 (Norm.norm (HSub.hSub x y))
    s : Finset E
    ⊢ Exists fun y => And (LE.le (Norm.norm y) R) (∀ (x : E), Membership.mem s x → …
  -/
  exact exists_norm_le_le_norm_sub_of_finset hc hR h s
  /-
    🎉 no goals
  -/


theorem exists_seq_norm_le_one_le_norm_sub (h : ¬FiniteDimensional 𝕜 E) :
    ∃ (R : ℝ) (f : ℕ → E), 1 < R ∧ (∀ n, ‖f n‖ ≤ R) ∧ Pairwise fun m n => 1 ≤ ‖f m - f n‖ := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    h : Not (FiniteDimensional 𝕜 E)
    ⊢ Exists fun R => Exists fun f => And (LT.lt 1 R) (And (∀ (n : Nat), LE.le (No …
  -/
  obtain ⟨c, hc⟩ : ∃ c : 𝕜, 1 < ‖c‖ := NormedField.exists_one_lt_norm 𝕜
  /-
    case intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    h : Not (FiniteDimensional 𝕜 E)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ⊢ Exists fun R => Exists fun f => And (LT.lt 1 R) (And (∀ (n : Nat), LE.le (No …
  -/
  have A : ‖c‖ < ‖c‖ + 1 := by linarith
  /-
    case intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    h : Not (FiniteDimensional 𝕜 E)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    A : LT.lt (Norm.norm c) (HAdd.hAdd (Norm.norm c) 1)
    ⊢ Exists fun R => Exists fun f => And (LT.lt 1 R) (And (∀ (n : Nat), LE.le (No …
  -/
  rcases exists_seq_norm_le_one_le_norm_sub' hc A h with ⟨f, hf⟩
  /-
    case intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    h : Not (FiniteDimensional 𝕜 E)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    A : LT.lt (Norm.norm c) (HAdd.hAdd (Norm.norm c) 1)
    f : Nat → E
    hf : And (∀ (n : Nat), LE.le (Norm.norm (f n)) (HAdd.hAdd (Norm.norm c) 1)) (P …
    ⊢ Exists fun R => Exists fun f => And (LT.lt 1 R) (And (∀ (n : Nat), LE.le (No …
  -/
  exact ⟨‖c‖ + 1, f, hc.trans A, hf.1, hf.2⟩
  /-
    🎉 no goals
  -/


/-- **Riesz's theorem**: if a closed ball with center zero of positive radius is compact in a vector
space, then the space is finite-dimensional. -/
theorem FiniteDimensional.of_isCompact_closedBall₀ {r : ℝ} (rpos : 0 < r)
    (h : IsCompact (Metric.closedBall (0 : E) r)) : FiniteDimensional 𝕜 E := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    h : IsCompact (Metric.closedBall 0 r)
    ⊢ FiniteDimensional 𝕜 E
  -/
  by_contra hfin
  obtain ⟨R, f, Rgt, fle, lef⟩ :
    ∃ (R : ℝ) (f : ℕ → E), 1 < R ∧ (∀ n, ‖f n‖ ≤ R) ∧ Pairwise fun m n => 1 ≤ ‖f m - f n‖ :=
    exists_seq_norm_le_one_le_norm_sub hfin
  /-
    case intro.intro.intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    h : IsCompact (Metric.closedBall 0 r)
    hfin : Not (FiniteDimensional 𝕜 E)
    R : Real
    f : Nat → E
    Rgt : LT.lt 1 R
    fle : ∀ (n : Nat), LE.le (Norm.norm (f n)) R
    lef : Pairwise fun m n => LE.le 1 (Norm.norm (HSub.hSub (f m) (f n)))
    ⊢ False
  -/
  have rRpos : 0 < r / R := div_pos rpos (zero_lt_one.trans Rgt)
  /-
    case intro.intro.intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    h : IsCompact (Metric.closedBall 0 r)
    hfin : Not (FiniteDimensional 𝕜 E)
    R : Real
    f : Nat → E
    Rgt : LT.lt 1 R
    fle : ∀ (n : Nat), LE.le (Norm.norm (f n)) R
    lef : Pairwise fun m n => LE.le 1 (Norm.norm (HSub.hSub (f m) (f n)))
    rRpos : LT.lt 0 (HDiv.hDiv r R)
    ⊢ False
  -/
  obtain ⟨c, hc⟩ : ∃ c : 𝕜, 0 < ‖c‖ ∧ ‖c‖ < r / R := NormedField.exists_norm_lt _ rRpos
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    h : IsCompact (Metric.closedBall 0 r)
    hfin : Not (FiniteDimensional 𝕜 E)
    R : Real
    f : Nat → E
    Rgt : LT.lt 1 R
    fle : ∀ (n : Nat), LE.le (Norm.norm (f n)) R
    lef : Pairwise fun m n => LE.le 1 (Norm.norm (HSub.hSub (f m) (f n)))
    rRpos : LT.lt 0 (HDiv.hDiv r R)
    c : 𝕜
    hc : And (LT.lt 0 (Norm.norm c)) (LT.lt (Norm.norm c) (HDiv.hDiv r R))
    ⊢ False
  -/
  let g := fun n : ℕ => c • f n
  have A : ∀ n, g n ∈ Metric.closedBall (0 : E) r := by
    intro n
    simp only [g, norm_smul, dist_zero_right, Metric.mem_closedBall]
    calc
      ‖c‖ * ‖f n‖ ≤ r / R * R := by
        gcongr
        · exact hc.2.le
        · apply fle
      _ = r := by field_simp [(zero_lt_one.trans Rgt).ne']
  -- Porting note: moved type ascriptions because of exists_prop changes
  obtain ⟨x : E, _ : x ∈ Metric.closedBall (0 : E) r, φ : ℕ → ℕ, φmono : StrictMono φ,
    φlim : Tendsto (g ∘ φ) atTop (𝓝 x)⟩ := h.tendsto_subseq A
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    h : IsCompact (Metric.closedBall 0 r)
    hfin : Not (FiniteDimensional 𝕜 E)
    R : Real
    f : Nat → E
    Rgt : LT.lt 1 R
    fle : ∀ (n : Nat), LE.le (Norm.norm (f n)) R
    lef : Pairwise fun m n => LE.le 1 (Norm.norm (HSub.hSub (f m) (f n)))
    rRpos : LT.lt 0 (HDiv.hDiv r R)
    c : 𝕜
    hc : And (LT.lt 0 (Norm.norm c)) (LT.lt (Norm.norm c) (HDiv.hDiv r R))
    g : Nat → E := fun n => HSMul.hSMul c (f n)
    A : ∀ (n : Nat), Membership.mem (Metric.closedBall 0 r) (g n)
    x : E
    left✝ : Membership.mem (Metric.closedBall 0 r) x
    φ : Nat → Nat
    φmono : StrictMono φ
    φlim : Filter.Tendsto (Function.comp g φ) Filter.atTop (nhds x)
    ⊢ False
  -/
  have B : CauchySeq (g ∘ φ) := φlim.cauchySeq
  obtain ⟨N, hN⟩ : ∃ N : ℕ, ∀ n : ℕ, N ≤ n → dist ((g ∘ φ) n) ((g ∘ φ) N) < ‖c‖ :=
    Metric.cauchySeq_iff'.1 B ‖c‖ hc.1
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace 𝕜
    r : Real
    rpos : LT.lt 0 r
    h : IsCompact (Metric.closedBall 0 r)
    hfin : Not (FiniteDimensional 𝕜 E)
    R : Real
    f : Nat → E
    Rgt : LT.lt 1 R
    fle : ∀ (n : Nat), LE.le (Norm.norm (f n)) R
    lef : Pairwise fun m n => LE.le 1 (Norm.norm (HSub.hSub (f m) (f n)))
    rRpos : LT.lt 0 (HDiv.hDiv r R)
    c : 𝕜
    hc : And (LT.lt 0 (Norm.norm c)) (LT.lt (Norm.norm c) (HDiv.hDiv r R))
    g : Nat → E := fun n => HSMul.hSMul c (f n)
    A : ∀ (n : Nat), Membership.mem (Metric.closedBall 0 r) (g n)
    x : E
    left✝ : Membership.mem (Metric.closedBall 0 r) x
    φ : Nat → Nat
    φmono : StrictMono φ
    φlim : Filter.Tendsto (Function.comp g φ) Filter.atTop (nhds x)
    B : CauchySeq (Function.comp g φ)
    N : Nat
    hN : ∀ (n : Nat), LE.le N n → LT.lt (Dist.dist (Function.comp g φ n) (Function …
    ⊢ False
  -/
  apply lt_irrefl ‖c‖
  calc
    ‖c‖ ≤ dist (g (φ (N + 1))) (g (φ N)) := by
      conv_lhs => rw [← mul_one ‖c‖]
      simp only [g, dist_eq_norm, ← smul_sub, norm_smul]
      gcongr
      apply lef (ne_of_gt _)
      exact φmono (Nat.lt_succ_self N)
    _ < ‖c‖ := hN (N + 1) (Nat.le_succ N)


@[deprecated (since := "2024-02-02")]
alias finiteDimensional_of_isCompact_closedBall₀ := FiniteDimensional.of_isCompact_closedBall₀


/-- **Riesz's theorem**: if a closed ball of positive radius is compact in a vector space, then the
space is finite-dimensional. -/
theorem FiniteDimensional.of_isCompact_closedBall {r : ℝ} (rpos : 0 < r) {c : E}
    (h : IsCompact (Metric.closedBall c r)) : FiniteDimensional 𝕜 E :=
                                         /-
                                           𝕜 : Type u
                                           inst✝³ : NontriviallyNormedField 𝕜
                                           E : Type v
                                           inst✝² : NormedAddCommGroup E
                                           inst✝¹ : NormedSpace 𝕜 E
                                           inst✝ : CompleteSpace 𝕜
                                           r : Real
                                           rpos : LT.lt 0 r
                                           c : E
                                           h : IsCompact (Metric.closedBall c r)
                                           ⊢ IsCompact (Metric.closedBall 0 r)
                                         -/
  .of_isCompact_closedBall₀ 𝕜 rpos <| by simpa using h.vadd (-c)
                                         /-
                                           🎉 no goals
                                         -/


@[deprecated (since := "2024-02-02")]
alias finiteDimensional_of_isCompact_closedBall := FiniteDimensional.of_isCompact_closedBall


/-- **Riesz's theorem**: a locally compact normed vector space is finite-dimensional. -/
theorem FiniteDimensional.of_locallyCompactSpace [LocallyCompactSpace E] :
    FiniteDimensional 𝕜 E :=
  let ⟨_r, rpos, hr⟩ := exists_isCompact_closedBall (0 : E)
  .of_isCompact_closedBall₀ 𝕜 rpos hr


@[deprecated (since := "2024-02-02")]
alias finiteDimensional_of_locallyCompactSpace := FiniteDimensional.of_locallyCompactSpace


/-- If a function has compact support, then either the function is trivial
or the space is finite-dimensional. -/
theorem HasCompactSupport.eq_zero_or_finiteDimensional {X : Type*} [TopologicalSpace X] [Zero X]
    [T1Space X] {f : E → X} (hf : HasCompactSupport f) (h'f : Continuous f) :
    f = 0 ∨ FiniteDimensional 𝕜 E :=
  (HasCompactSupport.eq_zero_or_locallyCompactSpace_of_addGroup hf h'f).imp_right fun h ↦
    -- TODO: Lean doesn't find the instance without this `have`
    have : LocallyCompactSpace E := h; .of_locallyCompactSpace 𝕜


/-- If a function has compact multiplicative support, then either the function is trivial
or the space is finite-dimensional. -/
@[to_additive existing]
theorem HasCompactMulSupport.eq_one_or_finiteDimensional {X : Type*} [TopologicalSpace X] [One X]
    [T1Space X] {f : E → X} (hf : HasCompactMulSupport f) (h'f : Continuous f) :
    f = 1 ∨ FiniteDimensional 𝕜 E :=
  have : T1Space (Additive X) := ‹_›
  HasCompactSupport.eq_zero_or_finiteDimensional (X := Additive X) 𝕜 hf h'f


/-- A locally compact normed vector space is proper. -/
lemma ProperSpace.of_locallyCompactSpace (𝕜 : Type*) [NontriviallyNormedField 𝕜]
    {E : Type*} [SeminormedAddCommGroup E] [NormedSpace 𝕜 E] [LocallyCompactSpace E] :
    ProperSpace E := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : LocallyCompactSpace E
    ⊢ ProperSpace E
  -/
  rcases exists_isCompact_closedBall (0 : E) with ⟨r, rpos, hr⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : LocallyCompactSpace E
    r : Real
    rpos : LT.lt 0 r
    hr : IsCompact (Metric.closedBall 0 r)
    ⊢ ProperSpace E
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  have hC : ∀ n, IsCompact (closedBall (0 : E) (‖c‖^n * r)) := fun n ↦ by
    have : c ^ n ≠ 0 := pow_ne_zero _ <| fun h ↦ by simp [h, zero_le_one.not_lt] at hc
    simpa [_root_.smul_closedBall' this] using hr.smul (c ^ n)
  have hTop : Tendsto (fun n ↦ ‖c‖^n * r) atTop atTop :=
    Tendsto.atTop_mul_const rpos (tendsto_pow_atTop_atTop_of_one_lt hc)
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : LocallyCompactSpace E
    r : Real
    rpos : LT.lt 0 r
    hr : IsCompact (Metric.closedBall 0 r)
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    hC : ∀ (n : Nat), IsCompact (Metric.closedBall 0 (HMul.hMul (HPow.hPow (Norm.n …
    hTop : Filter.Tendsto (fun n => HMul.hMul (HPow.hPow (Norm.norm c) n) r) Filte …
    ⊢ ProperSpace E
  -/
  exact .of_seq_closedBall hTop (Eventually.of_forall hC)
  /-
    🎉 no goals
  -/


lemma ProperSpace.of_locallyCompact_module [Nontrivial E] [LocallyCompactSpace E] :
    ProperSpace 𝕜 :=
  have : LocallyCompactSpace 𝕜 := by
    /-
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : CompleteSpace 𝕜
      inst✝¹ : Nontrivial E
      inst✝ : LocallyCompactSpace E
      ⊢ LocallyCompactSpace 𝕜
    -/
    obtain ⟨v, hv⟩ : ∃ v : E, v ≠ 0 := exists_ne 0
    /-
      case intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : CompleteSpace 𝕜
      inst✝¹ : Nontrivial E
      inst✝ : LocallyCompactSpace E
      v : E
      hv : Ne v 0
      ⊢ LocallyCompactSpace 𝕜
    -/
    let L : 𝕜 → E := fun t ↦ t • v
    /-
      case intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : CompleteSpace 𝕜
      inst✝¹ : Nontrivial E
      inst✝ : LocallyCompactSpace E
      v : E
      hv : Ne v 0
      L : 𝕜 → E := fun t => HSMul.hSMul t v
      ⊢ LocallyCompactSpace 𝕜
    -/
    have : IsClosedEmbedding L := isClosedEmbedding_smul_left hv
    /-
      case intro
      𝕜 : Type u
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type v
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : CompleteSpace 𝕜
      inst✝¹ : Nontrivial E
      inst✝ : LocallyCompactSpace E
      v : E
      hv : Ne v 0
      L : 𝕜 → E := fun t => HSMul.hSMul t v
      this : Topology.IsClosedEmbedding L
      ⊢ LocallyCompactSpace 𝕜
    -/
    apply IsClosedEmbedding.locallyCompactSpace this
    /-
      🎉 no goals
    -/
  .of_locallyCompactSpace 𝕜


/-- Continuous linear equivalence between continuous linear functions `𝕜ⁿ → E` and `Eⁿ`.
The spaces `𝕜ⁿ` and `Eⁿ` are represented as `ι → 𝕜` and `ι → E`, respectively,
where `ι` is a finite type. -/
def ContinuousLinearEquiv.piRing (ι : Type*) [Fintype ι] [DecidableEq ι] :
    ((ι → 𝕜) →L[𝕜] E) ≃L[𝕜] ι → E :=
  { LinearMap.toContinuousLinearMap.symm.trans (LinearEquiv.piRing 𝕜 E ι 𝕜) with
    continuous_toFun := by
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        ⊢ Continuous (↑__src✝).toFun
      -/
      refine continuous_pi fun i => ?_
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        i : ι
        ⊢ Continuous fun a => (↑__src✝).toFun a i
      -/
      exact (ContinuousLinearMap.apply 𝕜 E (Pi.single i 1)).continuous
      /-
        🎉 no goals
      -/
    continuous_invFun := by
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        ⊢ Continuous __src✝.invFun
      -/
      simp_rw [LinearEquiv.invFun_eq_symm, LinearEquiv.trans_symm, LinearEquiv.symm_symm]
      -- Note: added explicit type and removed `change` that tried to achieve the same
      refine AddMonoidHomClass.continuous_of_bound
        (LinearMap.toContinuousLinearMap.toLinearMap.comp
            (LinearEquiv.piRing 𝕜 E ι 𝕜).symm.toLinearMap)
        (Fintype.card ι : ℝ) fun g => ?_
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        g : ι → E
        ⊢ LE.le (Norm.norm (((↑LinearMap.toContinuousLinearMap).comp ↑(LinearEquiv.piR …
      -/
      rw [← nsmul_eq_mul]
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        g : ι → E
        ⊢ LE.le (Norm.norm (((↑LinearMap.toContinuousLinearMap).comp ↑(LinearEquiv.piR …
      -/
      refine opNorm_le_bound _ (nsmul_nonneg (norm_nonneg g) (Fintype.card ι)) fun t => ?_
      simp_rw [LinearMap.coe_comp, LinearEquiv.coe_toLinearMap, Function.comp_apply,
        LinearMap.coe_toContinuousLinearMap', LinearEquiv.piRing_symm_apply]
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        g : ι → E
        t : ι → 𝕜
        ⊢ LE.le (Norm.norm (Finset.univ.sum fun i => HSMul.hSMul (t i) (g i))) (HMul.h …
      -/
      apply le_trans (norm_sum_le _ _)
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        g : ι → E
        t : ι → 𝕜
        ⊢ LE.le (Finset.univ.sum fun i => Norm.norm (HSMul.hSMul (t i) (g i))) (HMul.h …
      -/
      rw [smul_mul_assoc]
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        g : ι → E
        t : ι → 𝕜
        ⊢ LE.le (Finset.univ.sum fun i => Norm.norm (HSMul.hSMul (t i) (g i))) (HSMul. …
      -/
      refine Finset.sum_le_card_nsmul _ _ _ fun i _ => ?_
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        g : ι → E
        t : ι → 𝕜
        i : ι
        x✝ : Membership.mem Finset.univ i
        ⊢ LE.le (Norm.norm (HSMul.hSMul (t i) (g i))) (HMul.hMul (Norm.norm g) (Norm.n …
      -/
      rw [norm_smul, mul_comm]
      /-
        𝕜 : Type u
        inst✝⁷ : NontriviallyNormedField 𝕜
        E : Type v
        inst✝⁶ : NormedAddCommGroup E
        inst✝⁵ : NormedSpace 𝕜 E
        F : Type w
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        inst✝² : CompleteSpace 𝕜
        ι : Type u_1
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        g : ι → E
        t : ι → 𝕜
        i : ι
        x✝ : Membership.mem Finset.univ i
        ⊢ LE.le (HMul.hMul (Norm.norm (g i)) (Norm.norm (t i))) (HMul.hMul (Norm.norm  …
      -/
                 /-
                   🎉 no goals
                 -/
      gcongr <;> apply norm_le_pi_norm }
                 /-
                   🎉 no goals
                 -/


/-- A family of continuous linear maps is continuous on `s` if all its applications are. -/
theorem continuousOn_clm_apply {X : Type*} [TopologicalSpace X] [FiniteDimensional 𝕜 E]
    {f : X → E →L[𝕜] F} {s : Set X} : ContinuousOn f s ↔ ∀ y, ContinuousOn (fun x => f x y) s := by
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FiniteDimensional 𝕜 E
    f : X → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set X
    ⊢ Iff (ContinuousOn f s) (∀ (y : E), ContinuousOn (fun x => (f x) y) s)
  -/
  refine ⟨fun h y => (ContinuousLinearMap.apply 𝕜 F y).continuous.comp_continuousOn h, fun h => ?_⟩
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FiniteDimensional 𝕜 E
    f : X → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set X
    h : ∀ (y : E), ContinuousOn (fun x => (f x) y) s
    ⊢ ContinuousOn f s
  -/
  let d := finrank 𝕜 E
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FiniteDimensional 𝕜 E
    f : X → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set X
    h : ∀ (y : E), ContinuousOn (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    ⊢ ContinuousOn f s
  -/
  have hd : d = finrank 𝕜 (Fin d → 𝕜) := (finrank_fin_fun 𝕜).symm
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FiniteDimensional 𝕜 E
    f : X → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set X
    h : ∀ (y : E), ContinuousOn (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    hd : Eq d (Module.finrank 𝕜 (Fin d → 𝕜))
    ⊢ ContinuousOn f s
  -/
  let e₁ : E ≃L[𝕜] Fin d → 𝕜 := ContinuousLinearEquiv.ofFinrankEq hd
  let e₂ : (E →L[𝕜] F) ≃L[𝕜] Fin d → F :=
    (e₁.arrowCongr (1 : F ≃L[𝕜] F)).trans (ContinuousLinearEquiv.piRing (Fin d))
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FiniteDimensional 𝕜 E
    f : X → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set X
    h : ∀ (y : E), ContinuousOn (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    hd : Eq d (Module.finrank 𝕜 (Fin d → 𝕜))
    e₁ : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin d → 𝕜) := ContinuousLinearEqu …
    e₂ : ContinuousLinearEquiv (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜)  …
    ⊢ ContinuousOn f s
  -/
  rw [← f.id_comp, ← e₂.symm_comp_self]
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FiniteDimensional 𝕜 E
    f : X → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set X
    h : ∀ (y : E), ContinuousOn (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    hd : Eq d (Module.finrank 𝕜 (Fin d → 𝕜))
    e₁ : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin d → 𝕜) := ContinuousLinearEqu …
    e₂ : ContinuousLinearEquiv (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜)  …
    ⊢ ContinuousOn (Function.comp (Function.comp ⇑e₂.symm ⇑e₂) f) s
  -/
  exact e₂.symm.continuous.comp_continuousOn (continuousOn_pi.mpr fun i => h _)
  /-
    🎉 no goals
  -/


theorem continuous_clm_apply {X : Type*} [TopologicalSpace X] [FiniteDimensional 𝕜 E]
    {f : X → E →L[𝕜] F} : Continuous f ↔ ∀ y, Continuous (f · y) := by
  /-
    𝕜 : Type u
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    F : Type w
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : CompleteSpace 𝕜
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : FiniteDimensional 𝕜 E
    f : X → ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Iff (Continuous f) (∀ (y : E), Continuous fun x => (f x) y)
  -/
  simp_rw [continuous_iff_continuousOn_univ, continuousOn_clm_apply]
  /-
    🎉 no goals
  -/


/-- Any finite-dimensional vector space over a locally compact field is proper.
We do not register this as an instance to avoid an instance loop when trying to prove the
properness of `𝕜`, and the search for `𝕜` as an unknown metavariable. Declare the instance
explicitly when needed. -/
theorem FiniteDimensional.proper [FiniteDimensional 𝕜 E] : ProperSpace E := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : LocallyCompactSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ ProperSpace E
  -/
  have : ProperSpace 𝕜 := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : LocallyCompactSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    this : ProperSpace 𝕜
    ⊢ ProperSpace E
  -/
  set e := ContinuousLinearEquiv.ofFinrankEq (@finrank_fin_fun 𝕜 _ _ (finrank 𝕜 E)).symm
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type v
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : LocallyCompactSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 E
    this : ProperSpace 𝕜
    e : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin (Module.finrank 𝕜 E) → 𝕜) := C …
    ⊢ ProperSpace E
  -/
  exact e.symm.antilipschitz.properSpace e.symm.continuous e.symm.surjective
  /-
    🎉 no goals
  -/


instance (priority := 900) FiniteDimensional.proper_real (E : Type u) [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] : ProperSpace E :=
  FiniteDimensional.proper ℝ E


/-- A submodule of a locally compact space over a complete field is also locally compact (and even
proper). -/
instance {𝕜 E : Type*} [NontriviallyNormedField 𝕜] [CompleteSpace 𝕜]
    [NormedAddCommGroup E] [NormedSpace 𝕜 E] [LocallyCompactSpace E] (S : Submodule 𝕜 E) :
    ProperSpace S := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : CompleteSpace 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : LocallyCompactSpace E
    S : Submodule 𝕜 E
    ⊢ ProperSpace (Subtype fun x => Membership.mem S x)
  -/
  nontriviality E
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : CompleteSpace 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : LocallyCompactSpace E
    S : Submodule 𝕜 E
    a✝ : Nontrivial E
    ⊢ ProperSpace (Subtype fun x => Membership.mem S x)
  -/
  have : ProperSpace 𝕜 := .of_locallyCompact_module 𝕜 E
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : CompleteSpace 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : LocallyCompactSpace E
    S : Submodule 𝕜 E
    a✝ : Nontrivial E
    this : ProperSpace 𝕜
    ⊢ ProperSpace (Subtype fun x => Membership.mem S x)
  -/
  have : FiniteDimensional 𝕜 E := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : CompleteSpace 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : LocallyCompactSpace E
    S : Submodule 𝕜 E
    a✝ : Nontrivial E
    this✝ : ProperSpace 𝕜
    this : FiniteDimensional 𝕜 E
    ⊢ ProperSpace (Subtype fun x => Membership.mem S x)
  -/
  exact FiniteDimensional.proper 𝕜 S
  /-
    🎉 no goals
  -/


/-- If `E` is a finite dimensional normed real vector space, `x : E`, and `s` is a neighborhood of
`x` that is not equal to the whole space, then there exists a point `y ∈ frontier s` at distance
`Metric.infDist x sᶜ` from `x`. See also
`IsCompact.exists_mem_frontier_infDist_compl_eq_dist`. -/
theorem exists_mem_frontier_infDist_compl_eq_dist {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] {x : E} {s : Set E} (hx : x ∈ s) (hs : s ≠ univ) :
    ∃ y ∈ frontier s, Metric.infDist x sᶜ = dist x y := by
  /-
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    x : E
    s : Set E
    hx : Membership.mem s x
    hs : Ne s Set.univ
    ⊢ Exists fun y => And (Membership.mem (frontier s) y) (Eq (Metric.infDist x (H …
  -/
  rcases Metric.exists_mem_closure_infDist_eq_dist (nonempty_compl.2 hs) x with ⟨y, hys, hyd⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    x : E
    s : Set E
    hx : Membership.mem s x
    hs : Ne s Set.univ
    y : E
    hys : Membership.mem (closure (HasCompl.compl s)) y
    hyd : Eq (Metric.infDist x (HasCompl.compl s)) (Dist.dist x y)
    ⊢ Exists fun y => And (Membership.mem (frontier s) y) (Eq (Metric.infDist x (H …
  -/
  rw [closure_compl] at hys
  refine ⟨y, ⟨Metric.closedBall_infDist_compl_subset_closure hx <|
    Metric.mem_closedBall.2 <| ge_of_eq ?_, hys⟩, hyd⟩
  /-
    case intro.intro
    E : Type u_1
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    x : E
    s : Set E
    hx : Membership.mem s x
    hs : Ne s Set.univ
    y : E
    hys : Membership.mem (HasCompl.compl (interior s)) y
    hyd : Eq (Metric.infDist x (HasCompl.compl s)) (Dist.dist x y)
    ⊢ Eq (Metric.infDist x (HasCompl.compl s)) (Dist.dist y x)
  -/
  rwa [dist_comm]
  /-
    🎉 no goals
  -/


/-- If `K` is a compact set in a nontrivial real normed space and `x ∈ K`, then there exists a point
`y` of the boundary of `K` at distance `Metric.infDist x Kᶜ` from `x`. See also
`exists_mem_frontier_infDist_compl_eq_dist`. -/
nonrec theorem IsCompact.exists_mem_frontier_infDist_compl_eq_dist {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [Nontrivial E] {x : E} {K : Set E} (hK : IsCompact K)
    (hx : x ∈ K) :
    ∃ y ∈ frontier K, Metric.infDist x Kᶜ = dist x y := by
  obtain hx' | hx' : x ∈ interior K ∪ frontier K := by
    rw [← closure_eq_interior_union_frontier]
    exact subset_closure hx
    /-
      case inl
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      K : Set E
      hK : IsCompact K
      hx : Membership.mem K x
      hx' : Membership.mem (interior K) x
      ⊢ Exists fun y => And (Membership.mem (frontier K) y) (Eq (Metric.infDist x (H …
    -/
  · rw [mem_interior_iff_mem_nhds, Metric.nhds_basis_closedBall.mem_iff] at hx'
    /-
      case inl
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      K : Set E
      hK : IsCompact K
      hx : Membership.mem K x
      hx' : Exists fun i => And (LT.lt 0 i) (HasSubset.Subset (Metric.closedBall x i …
      ⊢ Exists fun y => And (Membership.mem (frontier K) y) (Eq (Metric.infDist x (H …
    -/
    rcases hx' with ⟨r, hr₀, hrK⟩
    have : FiniteDimensional ℝ E :=
      .of_isCompact_closedBall ℝ hr₀
        (hK.of_isClosed_subset Metric.isClosed_ball hrK)
    /-
      case inl.intro.intro
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      K : Set E
      hK : IsCompact K
      hx : Membership.mem K x
      r : Real
      hr₀ : LT.lt 0 r
      hrK : HasSubset.Subset (Metric.closedBall x r) K
      this : FiniteDimensional Real E
      ⊢ Exists fun y => And (Membership.mem (frontier K) y) (Eq (Metric.infDist x (H …
    -/
    exact exists_mem_frontier_infDist_compl_eq_dist hx hK.ne_univ
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      K : Set E
      hK : IsCompact K
      hx : Membership.mem K x
      hx' : Membership.mem (frontier K) x
      ⊢ Exists fun y => And (Membership.mem (frontier K) y) (Eq (Metric.infDist x (H …
    -/
  · refine ⟨x, hx', ?_⟩
    /-
      case inr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      K : Set E
      hK : IsCompact K
      hx : Membership.mem K x
      hx' : Membership.mem (frontier K) x
      ⊢ Eq (Metric.infDist x (HasCompl.compl K)) (Dist.dist x x)
    -/
    rw [frontier_eq_closure_inter_closure] at hx'
    /-
      case inr
      E : Type u_1
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      inst✝ : Nontrivial E
      x : E
      K : Set E
      hK : IsCompact K
      hx : Membership.mem K x
      hx' : Membership.mem (Inter.inter (closure K) (closure (HasCompl.compl K))) x
      ⊢ Eq (Metric.infDist x (HasCompl.compl K)) (Dist.dist x x)
    -/
    rw [Metric.infDist_zero_of_mem_closure hx'.2, dist_self]
    /-
      🎉 no goals
    -/


/-- In a finite dimensional vector space over `ℝ`, the series `∑ x, ‖f x‖` is unconditionally
summable if and only if the series `∑ x, f x` is unconditionally summable. One implication holds in
any complete normed space, while the other holds only in finite dimensional spaces. -/
theorem summable_norm_iff {α E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {f : α → E} : (Summable fun x => ‖f x‖) ↔ Summable f := by
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    f : α → E
    ⊢ Iff (Summable fun x => Norm.norm (f x)) (Summable f)
  -/
  refine ⟨Summable.of_norm, fun hf ↦ ?_⟩
  -- First we use a finite basis to reduce the problem to the case `E = Fin N → ℝ`
  suffices ∀ {N : ℕ} {g : α → Fin N → ℝ}, Summable g → Summable fun x => ‖g x‖ by
    obtain v := Module.finBasis ℝ E
    set e := v.equivFunL
    have H : Summable fun x => ‖e (f x)‖ := this (e.summable.2 hf)
    refine .of_norm_bounded _ (H.mul_left ↑‖(e.symm : (Fin (finrank ℝ E) → ℝ) →L[ℝ] E)‖₊) fun i ↦ ?_
    simpa using (e.symm : (Fin (finrank ℝ E) → ℝ) →L[ℝ] E).le_opNorm (e <| f i)
  /-
    α : Type u_1
    E : Type u_2
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : FiniteDimensional Real E
    f : α → E
    hf : Summable f
    ⊢ ∀ {N : Nat} {g : α → Fin N → Real}, Summable g → Summable fun x => Norm.norm …
  -/
  clear! E
  -- Now we deal with `g : α → Fin N → ℝ`
  /-
    α : Type u_1
    ⊢ ∀ {N : Nat} {g : α → Fin N → Real}, Summable g → Summable fun x => Norm.norm …
  -/
  intro N g hg
  /-
    α : Type u_1
    N : Nat
    g : α → Fin N → Real
    hg : Summable g
    ⊢ Summable fun x => Norm.norm (g x)
  -/
  have : ∀ i, Summable fun x => ‖g x i‖ := fun i => (Pi.summable.1 hg i).abs
  /-
    α : Type u_1
    N : Nat
    g : α → Fin N → Real
    hg : Summable g
    this : ∀ (i : Fin N), Summable fun x => Norm.norm (g x i)
    ⊢ Summable fun x => Norm.norm (g x)
  -/
  refine .of_norm_bounded _ (summable_sum fun i (_ : i ∈ Finset.univ) => this i) fun x => ?_
  /-
    α : Type u_1
    N : Nat
    g : α → Fin N → Real
    hg : Summable g
    this : ∀ (i : Fin N), Summable fun x => Norm.norm (g x i)
    x : α
    ⊢ LE.le (Norm.norm (Norm.norm (g x))) (Finset.univ.sum fun i => Norm.norm (g x …
  -/
  rw [norm_norm, pi_norm_le_iff_of_nonneg]
    /-
      α : Type u_1
      N : Nat
      g : α → Fin N → Real
      hg : Summable g
      this : ∀ (i : Fin N), Summable fun x => Norm.norm (g x i)
      x : α
      ⊢ ∀ (i : Fin N), LE.le (Norm.norm (g x i)) (Finset.univ.sum fun i => Norm.norm …
    -/
  · refine fun i => Finset.single_le_sum (f := fun i => ‖g x i‖) (fun i _ => ?_) (Finset.mem_univ i)
    /-
      α : Type u_1
      N : Nat
      g : α → Fin N → Real
      hg : Summable g
      this : ∀ (i : Fin N), Summable fun x => Norm.norm (g x i)
      x : α
      i✝ i : Fin N
      x✝ : Membership.mem Finset.univ i
      ⊢ LE.le 0 ((fun i => Norm.norm (g x i)) i)
    -/
    exact norm_nonneg (g x i)
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      N : Nat
      g : α → Fin N → Real
      hg : Summable g
      this : ∀ (i : Fin N), Summable fun x => Norm.norm (g x i)
      x : α
      ⊢ LE.le 0 (Finset.univ.sum fun i => Norm.norm (g x i))
    -/
  · exact Finset.sum_nonneg fun _ _ => norm_nonneg _
    /-
      🎉 no goals
    -/


alias ⟨_, Summable.norm⟩ := summable_norm_iff


theorem summable_of_isBigO' {ι E F : Type*} [NormedAddCommGroup E] [CompleteSpace E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F] {f : ι → E} {g : ι → F}
    (hg : Summable g) (h : f =O[cofinite] g) : Summable f :=
  summable_of_isBigO hg.norm h.norm_right


lemma Asymptotics.IsBigO.comp_summable {ι E F : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    [NormedAddCommGroup F] [CompleteSpace F]
    {f : E → F} (hf : f =O[𝓝 0] id) {g : ι → E} (hg : Summable g) : Summable (f ∘ g) :=
  .of_norm <| hf.comp_summable_norm hg.norm


theorem summable_of_isBigO_nat' {E F : Type*} [NormedAddCommGroup E] [CompleteSpace E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] [FiniteDimensional ℝ F] {f : ℕ → E} {g : ℕ → F}
    (hg : Summable g) (h : f =O[atTop] g) : Summable f :=
  summable_of_isBigO_nat hg.norm h.norm_right


theorem summable_of_isEquivalent {ι E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {f : ι → E} {g : ι → E} (hg : Summable g) (h : f ~[cofinite] g) :
    Summable f :=
  hg.trans_sub (summable_of_isBigO' hg h.isLittleO.isBigO)


theorem summable_of_isEquivalent_nat {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {f : ℕ → E} {g : ℕ → E} (hg : Summable g) (h : f ~[atTop] g) :
    Summable f :=
  hg.trans_sub (summable_of_isBigO_nat' hg h.isLittleO.isBigO)


theorem IsEquivalent.summable_iff {ι E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {f : ι → E} {g : ι → E} (h : f ~[cofinite] g) :
    Summable f ↔ Summable g :=
  ⟨fun hf => summable_of_isEquivalent hf h.symm, fun hg => summable_of_isEquivalent hg h⟩


theorem IsEquivalent.summable_iff_nat {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {f : ℕ → E} {g : ℕ → E} (h : f ~[atTop] g) : Summable f ↔ Summable g :=
  ⟨fun hf => summable_of_isEquivalent_nat hf h.symm, fun hg => summable_of_isEquivalent_nat hg h⟩

