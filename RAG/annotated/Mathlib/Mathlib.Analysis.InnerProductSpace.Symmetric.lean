local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- A (not necessarily bounded) operator on an inner product space is symmetric, if for all
`x`, `y`, we have `⟪T x, y⟫ = ⟪x, T y⟫`. -/
def IsSymmetric (T : E →ₗ[𝕜] E) : Prop :=
  ∀ x y, ⟪T x, y⟫ = ⟪x, T y⟫


/-- An operator `T` on an inner product space is symmetric if and only if it is
`LinearMap.IsSelfAdjoint` with respect to the sesquilinear form given by the inner product. -/
theorem isSymmetric_iff_sesqForm (T : E →ₗ[𝕜] E) :
    T.IsSymmetric ↔ LinearMap.IsSelfAdjoint (R := 𝕜) (M := E) sesqFormOfInner T :=
  ⟨fun h x y => (h y x).symm, fun h x y => (h y x).symm⟩


theorem IsSymmetric.conj_inner_sym {T : E →ₗ[𝕜] E} (hT : IsSymmetric T) (x y : E) :
                                   /-
                                     𝕜 : Type u_1
                                     E : Type u_2
                                     inst✝² : RCLike 𝕜
                                     inst✝¹ : SeminormedAddCommGroup E
                                     inst✝ : InnerProductSpace 𝕜 E
                                     T : LinearMap (RingHom.id 𝕜) E E
                                     hT : T.IsSymmetric
                                     x y : E
                                     ⊢ Eq ((starRingEnd 𝕜) (Inner.inner (T x) y)) (Inner.inner (T y) x)
                                   -/
    conj ⟪T x, y⟫ = ⟪T y, x⟫ := by rw [hT x y, inner_conj_symm]
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem IsSymmetric.apply_clm {T : E →L[𝕜] E} (hT : IsSymmetric (T : E →ₗ[𝕜] E)) (x y : E) :
    ⟪T x, y⟫ = ⟪x, T y⟫ :=
  hT x y


@[simp]
protected theorem IsSymmetric.zero : (0 : E →ₗ[𝕜] E).IsSymmetric := fun x y =>
  (inner_zero_right x : ⟪x, 0⟫ = 0).symm ▸ (inner_zero_left y : ⟪0, y⟫ = 0)


@[deprecated (since := "2024-09-30")] alias isSymmetric_zero := IsSymmetric.zero


@[simp]
protected theorem IsSymmetric.id : (LinearMap.id : E →ₗ[𝕜] E).IsSymmetric := fun _ _ => rfl


@[deprecated (since := "2024-09-30")] alias isSymmetric_id := IsSymmetric.id


@[aesop safe apply]
theorem IsSymmetric.add {T S : E →ₗ[𝕜] E} (hT : T.IsSymmetric) (hS : S.IsSymmetric) :
    (T + S).IsSymmetric := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T S : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    hS : S.IsSymmetric
    ⊢ (HAdd.hAdd T S).IsSymmetric
  -/
  intro x y
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T S : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    hS : S.IsSymmetric
    x y : E
    ⊢ Eq (Inner.inner ((HAdd.hAdd T S) x) y) (Inner.inner x ((HAdd.hAdd T S) y))
  -/
  rw [add_apply, inner_add_left, hT x y, hS x y, ← inner_add_right, add_apply]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem IsSymmetric.sub {T S : E →ₗ[𝕜] E} (hT : T.IsSymmetric) (hS : S.IsSymmetric) :
    (T - S).IsSymmetric := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T S : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    hS : S.IsSymmetric
    ⊢ (HSub.hSub T S).IsSymmetric
  -/
  intro x y
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T S : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    hS : S.IsSymmetric
    x y : E
    ⊢ Eq (Inner.inner ((HSub.hSub T S) x) y) (Inner.inner x ((HSub.hSub T S) y))
  -/
  rw [sub_apply, inner_sub_left, hT x y, hS x y, ← inner_sub_right, sub_apply]
  /-
    🎉 no goals
  -/


@[aesop safe apply]
theorem IsSymmetric.smul {c : 𝕜} (hc : conj c = c) {T : E →ₗ[𝕜] E} (hT : T.IsSymmetric) :
    c • T |>.IsSymmetric := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    c : 𝕜
    hc : Eq ((starRingEnd 𝕜) c) c
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    ⊢ (HSMul.hSMul c T).IsSymmetric
  -/
  intro x y
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    c : 𝕜
    hc : Eq ((starRingEnd 𝕜) c) c
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    x y : E
    ⊢ Eq (Inner.inner ((HSMul.hSMul c T) x) y) (Inner.inner x ((HSMul.hSMul c T) y))
  -/
  simp only [smul_apply, inner_smul_left, hc, hT x y, inner_smul_right]
  /-
    🎉 no goals
  -/


@[aesop 30% apply]
lemma IsSymmetric.mul_of_commute {S T : E →ₗ[𝕜] E} (hS : S.IsSymmetric) (hT : T.IsSymmetric)
    (hST : Commute S T) : (S * T).IsSymmetric :=
               /-
                 𝕜 : Type u_1
                 E : Type u_2
                 inst✝² : RCLike 𝕜
                 inst✝¹ : SeminormedAddCommGroup E
                 inst✝ : InnerProductSpace 𝕜 E
                 S T : LinearMap (RingHom.id 𝕜) E E
                 hS : S.IsSymmetric
                 hT : T.IsSymmetric
                 hST : Commute S T
                 x✝¹ x✝ : E
                 ⊢ Eq (Inner.inner ((HMul.hMul S T) x✝¹) x✝) (Inner.inner x✝¹ ((HMul.hMul S T)  …
               -/
  fun _ _ ↦ by rw [mul_apply, hS, hT, hST, mul_apply]
               /-
                 🎉 no goals
               -/


@[aesop safe apply]
lemma IsSymmetric.pow {T : E →ₗ[𝕜] E} (hT : T.IsSymmetric) (n : ℕ) : (T ^ n).IsSymmetric := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    n : Nat
    ⊢ (HPow.hPow T n).IsSymmetric
  -/
  refine Nat.le_induction (by simp [one_eq_id]) (fun k _ ih ↦ ?_) n n.zero_le
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    n k : Nat
    x✝ : LE.le 0 k
    ih : (HPow.hPow T k).IsSymmetric
    ⊢ (HPow.hPow T (HAdd.hAdd k 1)).IsSymmetric
  -/
  rw [iterate_succ, ← mul_eq_comp]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    n k : Nat
    x✝ : LE.le 0 k
    ih : (HPow.hPow T k).IsSymmetric
    ⊢ LinearMap.IsSymmetric (HMul.hMul (HPow.hPow T k) T)
  -/
  exact ih.mul_of_commute hT <| .pow_left rfl k
  /-
    🎉 no goals
  -/


/-- For a symmetric operator `T`, the function `fun x ↦ ⟪T x, x⟫` is real-valued. -/
@[simp]
theorem IsSymmetric.coe_reApplyInnerSelf_apply {T : E →L[𝕜] E} (hT : IsSymmetric (T : E →ₗ[𝕜] E))
    (x : E) : (T.reApplyInnerSelf x : 𝕜) = ⟪T x, x⟫ := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : (↑T).IsSymmetric
    x : E
    ⊢ Eq (↑(T.reApplyInnerSelf x)) (Inner.inner (T x) x)
  -/
  rsuffices ⟨r, hr⟩ : ∃ r : ℝ, ⟪T x, x⟫ = r
    /-
      case intro
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : ContinuousLinearMap (RingHom.id 𝕜) E E
      hT : (↑T).IsSymmetric
      x : E
      r : Real
      hr : Eq (Inner.inner (T x) x) ↑r
      ⊢ Eq (↑(T.reApplyInnerSelf x)) (Inner.inner (T x) x)
    -/
  · simp [hr, T.reApplyInnerSelf_apply]
    /-
      🎉 no goals
    -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : (↑T).IsSymmetric
    x : E
    ⊢ Exists fun r => Eq (Inner.inner (T x) x) ↑r
  -/
  rw [← conj_eq_iff_real]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : (↑T).IsSymmetric
    x : E
    ⊢ Eq ((starRingEnd 𝕜) (Inner.inner (T x) x)) (Inner.inner (T x) x)
  -/
  exact hT.conj_inner_sym x x
  /-
    🎉 no goals
  -/


/-- If a symmetric operator preserves a submodule, its restriction to that submodule is
symmetric. -/
theorem IsSymmetric.restrict_invariant {T : E →ₗ[𝕜] E} (hT : IsSymmetric T) {V : Submodule 𝕜 E}
    (hV : ∀ v ∈ V, T v ∈ V) : IsSymmetric (T.restrict hV) := fun v w => hT v w


theorem IsSymmetric.restrictScalars {T : E →ₗ[𝕜] E} (hT : T.IsSymmetric) :
    letI := InnerProductSpace.rclikeToReal 𝕜 E
    letI : IsScalarTower ℝ 𝕜 E := RestrictScalars.isScalarTower _ _ _
    (T.restrictScalars ℝ).IsSymmetric :=
                /-
                  𝕜 : Type u_1
                  E : Type u_2
                  inst✝² : RCLike 𝕜
                  inst✝¹ : SeminormedAddCommGroup E
                  inst✝ : InnerProductSpace 𝕜 E
                  T : LinearMap (RingHom.id 𝕜) E E
                  hT : T.IsSymmetric
                  x y : E
                  ⊢ Eq (Inner.inner ((↑Real T) x) y) (Inner.inner x ((↑Real T) y))
                -/
  fun x y => by simp [hT x y, real_inner_eq_re_inner, LinearMap.coe_restrictScalars ℝ]
                /-
                  🎉 no goals
                -/


attribute [local simp] map_ofNat in -- use `ofNat` simp theorem with bad keys
open scoped InnerProductSpace in
/-- A linear operator on a complex inner product space is symmetric precisely when
`⟪T v, v⟫_ℂ` is real for all v. -/
theorem isSymmetric_iff_inner_map_self_real (T : V →ₗ[ℂ] V) :
    IsSymmetric T ↔ ∀ v : V, conj ⟪T v, v⟫_ℂ = ⟪T v, v⟫_ℂ := by
  /-
    V : Type u_3
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : InnerProductSpace Complex V
    T : LinearMap (RingHom.id Complex) V V
    ⊢ Iff T.IsSymmetric (∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v …
  -/
  constructor
    /-
      case mp
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      ⊢ T.IsSymmetric → ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v))  …
    -/
  · intro hT v
    /-
      case mp
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      hT : T.IsSymmetric
      v : V
      ⊢ Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner (T v) v)
    -/
    apply IsSymmetric.conj_inner_sym hT
    /-
      🎉 no goals
    -/
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      ⊢ (∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner (T …
    -/
  · intro h x y
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (Inner.inner (T x) y) (Inner.inner x (T y))
    -/
    rw [← inner_conj_symm x (T y)]
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (Inner.inner (T x) y) ((starRingEnd Complex) (Inner.inner (T y) x))
    -/
    rw [inner_map_polarization T x y]
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (Inner.inner (T x) y) ((starRingEnd Complex) (HDiv.hDiv (HSub.hSub (HAdd. …
    -/
    simp only [starRingEnd_apply, star_div₀, star_sub, star_add, star_mul]
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (Inner.inner (T x) y) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (Star.s …
    -/
    simp only [← starRingEnd_apply]
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (Inner.inner (T x) y) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub ((starR …
    -/
    rw [h (x + y), h (x - y), h (x + Complex.I • y), h (x - Complex.I • y)]
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (Inner.inner (T x) y) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (Inner. …
    -/
    simp only [Complex.conj_I]
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (Inner.inner (T x) y) (HDiv.hDiv (HSub.hSub (HAdd.hAdd (HSub.hSub (Inner. …
    -/
    rw [inner_map_polarization']
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HSub.hSub (HSub.hSub (Inner.inner (T (HAdd.hAdd x  …
    -/
    norm_num
    /-
      case mpr
      V : Type u_3
      inst✝¹ : SeminormedAddCommGroup V
      inst✝ : InnerProductSpace Complex V
      T : LinearMap (RingHom.id Complex) V V
      h : ∀ (v : V), Eq ((starRingEnd Complex) (Inner.inner (T v) v)) (Inner.inner ( …
      x y : V
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HSub.hSub (HSub.hSub (Inner.inner (HAdd.hAdd (T x) …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- Polarization identity for symmetric linear maps.
See `inner_map_polarization` for the complex version without the symmetric assumption. -/
theorem IsSymmetric.inner_map_polarization {T : E →ₗ[𝕜] E} (hT : T.IsSymmetric) (x y : E) :
    ⟪T x, y⟫ =
      (⟪T (x + y), x + y⟫ - ⟪T (x - y), x - y⟫ - I * ⟪T (x + (I : 𝕜) • y), x + (I : 𝕜) • y⟫ +
          I * ⟪T (x - (I : 𝕜) • y), x - (I : 𝕜) • y⟫) /
        4 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    x y : E
    ⊢ Eq (Inner.inner (T x) y) (HDiv.hDiv (HAdd.hAdd (HSub.hSub (HSub.hSub (Inner. …
  -/
  rcases@I_mul_I_ax 𝕜 _ with (h | h)
  · simp_rw [h, zero_mul, sub_zero, add_zero, map_add, map_sub, inner_add_left,
      inner_add_right, inner_sub_left, inner_sub_right, hT x, ← inner_conj_symm x (T y)]
    suffices (re ⟪T y, x⟫ : 𝕜) = ⟪T y, x⟫ by
      rw [conj_eq_iff_re.mpr this]
      ring
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : LinearMap (RingHom.id 𝕜) E E
      hT : T.IsSymmetric
      x y : E
      h : Eq RCLike.I 0
      ⊢ Eq (↑(RCLike.re (Inner.inner (T y) x))) (Inner.inner (T y) x)
    -/
    rw [← re_add_im ⟪T y, x⟫]
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : LinearMap (RingHom.id 𝕜) E E
      hT : T.IsSymmetric
      x y : E
      h : Eq RCLike.I 0
      ⊢ Eq (↑(RCLike.re (HAdd.hAdd (↑(RCLike.re (Inner.inner (T y) x))) (HMul.hMul ( …
    -/
    simp_rw [h, mul_zero, add_zero]
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : LinearMap (RingHom.id 𝕜) E E
      hT : T.IsSymmetric
      x y : E
      h : Eq RCLike.I 0
      ⊢ Eq ↑(RCLike.re ↑(RCLike.re (Inner.inner (T y) x))) ↑(RCLike.re (Inner.inner  …
    -/
    norm_cast
    /-
      🎉 no goals
    -/
  · simp_rw [map_add, map_sub, inner_add_left, inner_add_right, inner_sub_left, inner_sub_right,
      LinearMap.map_smul, inner_smul_left, inner_smul_right, RCLike.conj_I, mul_add, mul_sub,
      sub_sub, ← mul_assoc, mul_neg, h, neg_neg, one_mul, neg_one_mul]
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      inst✝² : RCLike 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : InnerProductSpace 𝕜 E
      T : LinearMap (RingHom.id 𝕜) E E
      hT : T.IsSymmetric
      x y : E
      h : Eq (HMul.hMul RCLike.I RCLike.I) (-1)
      ⊢ Eq (Inner.inner (T x) y) (HDiv.hDiv (HAdd.hAdd (HSub.hSub (HAdd.hAdd (HAdd.h …
    -/
    ring
    /-
      🎉 no goals
    -/


/-- The **Hellinger--Toeplitz theorem**: if a symmetric operator is defined on a complete space,
  then it is automatically continuous. -/
theorem IsSymmetric.continuous [CompleteSpace E] {T : E →ₗ[𝕜] E} (hT : IsSymmetric T) :
    Continuous T := by
  -- We prove it by using the closed graph theorem
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    ⊢ Continuous ⇑T
  -/
  refine T.continuous_of_seq_closed_graph fun u x y hu hTu => ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    u : Nat → E
    x y : E
    hu : Filter.Tendsto u Filter.atTop (nhds x)
    hTu : Filter.Tendsto (Function.comp (⇑T) u) Filter.atTop (nhds y)
    ⊢ Eq y (T x)
  -/
  rw [← sub_eq_zero, ← @inner_self_eq_zero 𝕜]
  have hlhs : ∀ k : ℕ, ⟪T (u k) - T x, y - T x⟫ = ⟪u k - x, T (y - T x)⟫ := by
    intro k
    rw [← T.map_sub, hT]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    u : Nat → E
    x y : E
    hu : Filter.Tendsto u Filter.atTop (nhds x)
    hTu : Filter.Tendsto (Function.comp (⇑T) u) Filter.atTop (nhds y)
    hlhs : ∀ (k : Nat), Eq (Inner.inner (HSub.hSub (T (u k)) (T x)) (HSub.hSub y ( …
    ⊢ Eq (Inner.inner (HSub.hSub y (T x)) (HSub.hSub y (T x))) 0
  -/
  refine tendsto_nhds_unique ((hTu.sub_const _).inner tendsto_const_nhds) ?_
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    u : Nat → E
    x y : E
    hu : Filter.Tendsto u Filter.atTop (nhds x)
    hTu : Filter.Tendsto (Function.comp (⇑T) u) Filter.atTop (nhds y)
    hlhs : ∀ (k : Nat), Eq (Inner.inner (HSub.hSub (T (u k)) (T x)) (HSub.hSub y ( …
    ⊢ Filter.Tendsto (fun t => Inner.inner (HSub.hSub (Function.comp (⇑T) u t) (T  …
  -/
  simp_rw [Function.comp_apply, hlhs]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    u : Nat → E
    x y : E
    hu : Filter.Tendsto u Filter.atTop (nhds x)
    hTu : Filter.Tendsto (Function.comp (⇑T) u) Filter.atTop (nhds y)
    hlhs : ∀ (k : Nat), Eq (Inner.inner (HSub.hSub (T (u k)) (T x)) (HSub.hSub y ( …
    ⊢ Filter.Tendsto (fun t => Inner.inner (HSub.hSub (u t) x) (T (HSub.hSub y (T  …
  -/
  rw [← inner_zero_left (T (y - T x))]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    u : Nat → E
    x y : E
    hu : Filter.Tendsto u Filter.atTop (nhds x)
    hTu : Filter.Tendsto (Function.comp (⇑T) u) Filter.atTop (nhds y)
    hlhs : ∀ (k : Nat), Eq (Inner.inner (HSub.hSub (T (u k)) (T x)) (HSub.hSub y ( …
    ⊢ Filter.Tendsto (fun t => Inner.inner (HSub.hSub (u t) x) (T (HSub.hSub y (T  …
  -/
  refine Filter.Tendsto.inner ?_ tendsto_const_nhds
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    u : Nat → E
    x y : E
    hu : Filter.Tendsto u Filter.atTop (nhds x)
    hTu : Filter.Tendsto (Function.comp (⇑T) u) Filter.atTop (nhds y)
    hlhs : ∀ (k : Nat), Eq (Inner.inner (HSub.hSub (T (u k)) (T x)) (HSub.hSub y ( …
    ⊢ Filter.Tendsto (fun t => HSub.hSub (u t) x) Filter.atTop (nhds 0)
  -/
  rw [← sub_self x]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    u : Nat → E
    x y : E
    hu : Filter.Tendsto u Filter.atTop (nhds x)
    hTu : Filter.Tendsto (Function.comp (⇑T) u) Filter.atTop (nhds y)
    hlhs : ∀ (k : Nat), Eq (Inner.inner (HSub.hSub (T (u k)) (T x)) (HSub.hSub y ( …
    ⊢ Filter.Tendsto (fun t => HSub.hSub (u t) x) Filter.atTop (nhds (HSub.hSub x  …
  -/
  exact hu.sub_const _
  /-
    🎉 no goals
  -/


/-- A symmetric linear map `T` is zero if and only if `⟪T x, x⟫_ℝ = 0` for all `x`.
See `inner_map_self_eq_zero` for the complex version without the symmetric assumption. -/
theorem IsSymmetric.inner_map_self_eq_zero {T : E →ₗ[𝕜] E} (hT : T.IsSymmetric) :
    (∀ x, ⟪T x, x⟫ = 0) ↔ T = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    ⊢ Iff (∀ (x : E), Eq (Inner.inner (T x) x) 0) (Eq T 0)
  -/
  simp_rw [LinearMap.ext_iff, zero_apply]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    ⊢ Iff (∀ (x : E), Eq (Inner.inner (T x) x) 0) (∀ (x : E), Eq (T x) 0)
  -/
  refine ⟨fun h x => ?_, fun h => by simp_rw [h, inner_zero_left, forall_const]⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    h : ∀ (x : E), Eq (Inner.inner (T x) x) 0
    x : E
    ⊢ Eq (T x) 0
  -/
  rw [← @inner_self_eq_zero 𝕜, hT.inner_map_polarization]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    h : ∀ (x : E), Eq (Inner.inner (T x) x) 0
    x : E
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HSub.hSub (HSub.hSub (Inner.inner (T (HAdd.hAdd x  …
  -/
  simp_rw [h _]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace 𝕜 E
    T : LinearMap (RingHom.id 𝕜) E E
    hT : T.IsSymmetric
    h : ∀ (x : E), Eq (Inner.inner (T x) x) 0
    x : E
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HSub.hSub (HSub.hSub 0 0) (HMul.hMul RCLike.I 0))  …
  -/
  ring
  /-
    🎉 no goals
  -/


