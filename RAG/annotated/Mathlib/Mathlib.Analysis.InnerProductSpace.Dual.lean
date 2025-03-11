local notation "⟪" x ", " y "⟫" => @inner 𝕜 E _ x y


local postfix:90 "†" => starRingEnd _


/-- An element `x` of an inner product space `E` induces an element of the dual space `Dual 𝕜 E`,
the map `fun y => ⟪x, y⟫`; moreover this operation is a conjugate-linear isometric embedding of `E`
into `Dual 𝕜 E`.
If `E` is complete, this operation is surjective, hence a conjugate-linear isometric equivalence;
see `toDual`.
-/
def toDualMap : E →ₗᵢ⋆[𝕜] NormedSpace.Dual 𝕜 E :=
  { innerSL 𝕜 with norm_map' := innerSL_apply_norm _ }


@[simp]
theorem toDualMap_apply {x y : E} : toDualMap 𝕜 E x y = ⟪x, y⟫ :=
  rfl


/-- For each `x : E`, the kernel of `⟪x, ⬝⟫` includes the null space. -/
lemma nullSubmodule_le_ker_toDualMap_right (x : E) : nullSubmodule 𝕜 E ≤ ker (toDualMap 𝕜 E x) :=
  fun _ hx ↦ inner_eq_zero_of_right x ((mem_nullSubmodule_iff).mp hx)


/-- The kernel of the map `x ↦ ⟪·, x⟫` includes the null space. -/
lemma nullSubmodule_le_ker_toDualMap_left : nullSubmodule 𝕜 E ≤ ker (toDualMap 𝕜 E) :=
  fun _ hx ↦ ContinuousLinearMap.ext <| fun y ↦ inner_eq_zero_of_left y hx


theorem innerSL_norm [Nontrivial E] : ‖(innerSL 𝕜 : E →L⋆[𝕜] E →L[𝕜] 𝕜)‖ = 1 :=
  show ‖(toDualMap 𝕜 E).toContinuousLinearMap‖ = 1 from LinearIsometry.norm_toContinuousLinearMap _


theorem ext_inner_left_basis {ι : Type*} {x y : E} (b : Basis ι 𝕜 E)
    (h : ∀ i : ι, ⟪b i, x⟫ = ⟪b i, y⟫) : x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    ⊢ Eq x y
  -/
  apply (toDualMap 𝕜 E).map_eq_iff.mp
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    ⊢ Eq ((InnerProductSpace.toDualMap 𝕜 E) x) ((InnerProductSpace.toDualMap 𝕜 E) y)
  -/
  refine (Function.Injective.eq_iff ContinuousLinearMap.coe_injective).mp (Basis.ext b ?_)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    ⊢ ∀ (i : ι), Eq (↑((InnerProductSpace.toDualMap 𝕜 E) x) (b i)) (↑((InnerProduc …
  -/
  intro i
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    i : ι
    ⊢ Eq (↑((InnerProductSpace.toDualMap 𝕜 E) x) (b i)) (↑((InnerProductSpace.toDu …
  -/
  simp only [ContinuousLinearMap.coe_coe]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    i : ι
    ⊢ Eq (((InnerProductSpace.toDualMap 𝕜 E) x) (b i)) (((InnerProductSpace.toDual …
  -/
  rw [toDualMap_apply, toDualMap_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    i : ι
    ⊢ Eq (Inner.inner x (b i)) (Inner.inner y (b i))
  -/
  rw [← inner_conj_symm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    i : ι
    ⊢ Eq ((starRingEnd 𝕜) (Inner.inner (b i) x)) (Inner.inner y (b i))
  -/
  conv_rhs => rw [← inner_conj_symm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
    i : ι
    ⊢ Eq ((starRingEnd 𝕜) (Inner.inner (b i) x)) ((starRingEnd 𝕜) (Inner.inner (b  …
  -/
  exact congr_arg conj (h i)
  /-
    🎉 no goals
  -/


theorem ext_inner_right_basis {ι : Type*} {x y : E} (b : Basis ι 𝕜 E)
    (h : ∀ i : ι, ⟪x, b i⟫ = ⟪y, b i⟫) : x = y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner x (b i)) (Inner.inner y (b i))
    ⊢ Eq x y
  -/
  refine ext_inner_left_basis b fun i => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner x (b i)) (Inner.inner y (b i))
    i : ι
    ⊢ Eq (Inner.inner (b i) x) (Inner.inner (b i) y)
  -/
  rw [← inner_conj_symm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner x (b i)) (Inner.inner y (b i))
    i : ι
    ⊢ Eq ((starRingEnd 𝕜) (Inner.inner x (b i))) (Inner.inner (b i) y)
  -/
  conv_rhs => rw [← inner_conj_symm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    ι : Type u_3
    x y : E
    b : Basis ι 𝕜 E
    h : ∀ (i : ι), Eq (Inner.inner x (b i)) (Inner.inner y (b i))
    i : ι
    ⊢ Eq ((starRingEnd 𝕜) (Inner.inner x (b i))) ((starRingEnd 𝕜) (Inner.inner y ( …
  -/
  exact congr_arg conj (h i)
  /-
    🎉 no goals
  -/


/-- Fréchet-Riesz representation: any `ℓ` in the dual of a Hilbert space `E` is of the form
`fun u => ⟪y, u⟫` for some `y : E`, i.e. `toDualMap` is surjective.
-/
def toDual : E ≃ₗᵢ⋆[𝕜] NormedSpace.Dual 𝕜 E :=
  LinearIsometryEquiv.ofSurjective (toDualMap 𝕜 E)
    (by
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        inst✝ : CompleteSpace E
        ⊢ Function.Surjective ⇑(InnerProductSpace.toDualMap 𝕜 E)
      -/
      intro ℓ
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        inst✝ : CompleteSpace E
        ℓ : NormedSpace.Dual 𝕜 E
        ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
      -/
      set Y := LinearMap.ker ℓ
      /-
        𝕜 : Type u_1
        E : Type u_2
        inst✝³ : RCLike 𝕜
        inst✝² : NormedAddCommGroup E
        inst✝¹ : InnerProductSpace 𝕜 E
        inst✝ : CompleteSpace E
        ℓ : NormedSpace.Dual 𝕜 E
        Y : Submodule 𝕜 E := LinearMap.ker ℓ
        ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
      -/
      by_cases htriv : Y = ⊤
      · have hℓ : ℓ = 0 := by
          have h' := LinearMap.ker_eq_top.mp htriv
          rw [← coe_zero] at h'
          apply coe_injective
          exact h'
        /-
          case pos
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          htriv : Eq Y Top.top
          hℓ : Eq ℓ 0
          ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
        -/
        exact ⟨0, by simp [hℓ]⟩
        /-
          🎉 no goals
        -/
        /-
          case neg
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          htriv : Not (Eq Y Top.top)
          ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
        -/
      · rw [← Submodule.orthogonal_eq_bot_iff] at htriv
        /-
          case neg
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          htriv : Not (Eq Y.orthogonal Bot.bot)
          ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
        -/
        change Yᗮ ≠ ⊥ at htriv
        /-
          case neg
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          htriv : Ne Y.orthogonal Bot.bot
          ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
        -/
        rw [Submodule.ne_bot_iff] at htriv
        /-
          case neg
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          htriv : Exists fun x => And (Membership.mem Y.orthogonal x) (Ne x 0)
          ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
        -/
        obtain ⟨z : E, hz : z ∈ Yᗮ, z_ne_0 : z ≠ 0⟩ := htriv
        /-
          case neg.intro.intro
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          z : E
          hz : Membership.mem Y.orthogonal z
          z_ne_0 : Ne z 0
          ⊢ Exists fun a => Eq ((InnerProductSpace.toDualMap 𝕜 E) a) ℓ
        -/
        refine ⟨(starRingEnd (R := 𝕜) (ℓ z) / ⟪z, z⟫) • z, ?_⟩
        /-
          case neg.intro.intro
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          z : E
          hz : Membership.mem Y.orthogonal z
          z_ne_0 : Ne z 0
          ⊢ Eq ((InnerProductSpace.toDualMap 𝕜 E) (HSMul.hSMul (HDiv.hDiv ((starRingEnd  …
        -/
        apply ContinuousLinearMap.ext
        /-
          case neg.intro.intro.h
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          z : E
          hz : Membership.mem Y.orthogonal z
          z_ne_0 : Ne z 0
          ⊢ ∀ (x : E), Eq (((InnerProductSpace.toDualMap 𝕜 E) (HSMul.hSMul (HDiv.hDiv (( …
        -/
        intro x
        have h₁ : ℓ z • x - ℓ x • z ∈ Y := by
          rw [LinearMap.mem_ker, map_sub, ContinuousLinearMap.map_smul,
            ContinuousLinearMap.map_smul, Algebra.id.smul_eq_mul, Algebra.id.smul_eq_mul, mul_comm]
          exact sub_self (ℓ x * ℓ z)
        have h₂ : ℓ z * ⟪z, x⟫ = ℓ x * ⟪z, z⟫ :=
          haveI h₃ :=
            calc
              0 = ⟪z, ℓ z • x - ℓ x • z⟫ := by
                rw [(Y.mem_orthogonal' z).mp hz]
                exact h₁
              _ = ⟪z, ℓ z • x⟫ - ⟪z, ℓ x • z⟫ := by rw [inner_sub_right]
              _ = ℓ z * ⟪z, x⟫ - ℓ x * ⟪z, z⟫ := by simp [inner_smul_right]
          sub_eq_zero.mp (Eq.symm h₃)
        have h₄ :=
          calc
            ⟪(ℓ z† / ⟪z, z⟫) • z, x⟫ = ℓ z / ⟪z, z⟫ * ⟪z, x⟫ := by simp [inner_smul_left, conj_conj]
            _ = ℓ z * ⟪z, x⟫ / ⟪z, z⟫ := by rw [← div_mul_eq_mul_div]
            _ = ℓ x * ⟪z, z⟫ / ⟪z, z⟫ := by rw [h₂]
            _ = ℓ x := by field_simp [inner_self_ne_zero.2 z_ne_0]
        /-
          case neg.intro.intro.h
          𝕜 : Type u_1
          E : Type u_2
          inst✝³ : RCLike 𝕜
          inst✝² : NormedAddCommGroup E
          inst✝¹ : InnerProductSpace 𝕜 E
          inst✝ : CompleteSpace E
          ℓ : NormedSpace.Dual 𝕜 E
          Y : Submodule 𝕜 E := LinearMap.ker ℓ
          z : E
          hz : Membership.mem Y.orthogonal z
          z_ne_0 : Ne z 0
          x : E
          h₁ : Membership.mem Y (HSub.hSub (HSMul.hSMul (ℓ z) x) (HSMul.hSMul (ℓ x) z))
          h₂ : Eq (HMul.hMul (ℓ z) (Inner.inner z x)) (HMul.hMul (ℓ x) (Inner.inner z z))
          h₄ : Eq (Inner.inner (HSMul.hSMul (HDiv.hDiv ((starRingEnd 𝕜) (ℓ z)) (Inner.in …
          ⊢ Eq (((InnerProductSpace.toDualMap 𝕜 E) (HSMul.hSMul (HDiv.hDiv ((starRingEnd …
        -/
        exact h₄)
        /-
          🎉 no goals
        -/


@[simp]
theorem toDual_apply {x y : E} : toDual 𝕜 E x y = ⟪x, y⟫ :=
  rfl


@[simp]
theorem toDual_symm_apply {x : E} {y : NormedSpace.Dual 𝕜 E} : ⟪(toDual 𝕜 E).symm y, x⟫ = y x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    x : E
    y : NormedSpace.Dual 𝕜 E
    ⊢ Eq (Inner.inner ((InnerProductSpace.toDual 𝕜 E).symm y) x) (y x)
  -/
  rw [← toDual_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    x : E
    y : NormedSpace.Dual 𝕜 E
    ⊢ Eq (((InnerProductSpace.toDual 𝕜 E) ((InnerProductSpace.toDual 𝕜 E).symm y)) …
  -/
  simp only [LinearIsometryEquiv.apply_symm_apply]
  /-
    🎉 no goals
  -/


/-- Maps a bounded sesquilinear form to its continuous linear map,
given by interpreting the form as a map `B : E →L⋆[𝕜] NormedSpace.Dual 𝕜 E`
and dualizing the result using `toDual`.
-/
def continuousLinearMapOfBilin (B : E →L⋆[𝕜] E →L[𝕜] 𝕜) : E →L[𝕜] E :=
  comp (toDual 𝕜 E).symm.toContinuousLinearEquiv.toContinuousLinearMap B


local postfix:1024 "♯" => continuousLinearMapOfBilin


@[simp]
theorem continuousLinearMapOfBilin_apply (v w : E) : ⟪B♯ v, w⟫ = B v w := by
  rw [continuousLinearMapOfBilin, coe_comp', ContinuousLinearEquiv.coe_coe,
    LinearIsometryEquiv.coe_toContinuousLinearEquiv, Function.comp_apply, toDual_symm_apply]


theorem unique_continuousLinearMapOfBilin {v f : E} (is_lax_milgram : ∀ w, ⟪f, w⟫ = B v w) :
    f = B♯ v := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    B : ContinuousLinearMap (starRingEnd 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    v f : E
    is_lax_milgram : ∀ (w : E), Eq (Inner.inner f w) ((B v) w)
    ⊢ Eq f ((InnerProductSpace.continuousLinearMapOfBilin B) v)
  -/
  refine ext_inner_right 𝕜 ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    B : ContinuousLinearMap (starRingEnd 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    v f : E
    is_lax_milgram : ∀ (w : E), Eq (Inner.inner f w) ((B v) w)
    ⊢ ∀ (v_1 : E), Eq (Inner.inner f v_1) (Inner.inner ((InnerProductSpace.continu …
  -/
  intro w
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    B : ContinuousLinearMap (starRingEnd 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    v f : E
    is_lax_milgram : ∀ (w : E), Eq (Inner.inner f w) ((B v) w)
    w : E
    ⊢ Eq (Inner.inner f w) (Inner.inner ((InnerProductSpace.continuousLinearMapOfB …
  -/
  rw [continuousLinearMapOfBilin_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    B : ContinuousLinearMap (starRingEnd 𝕜) E (ContinuousLinearMap (RingHom.id 𝕜)  …
    v f : E
    is_lax_milgram : ∀ (w : E), Eq (Inner.inner f w) ((B v) w)
    w : E
    ⊢ Eq (Inner.inner f w) ((B v) w)
  -/
  exact is_lax_milgram w
  /-
    🎉 no goals
  -/


