local notation "⟪" x ", " y "⟫" => @inner 𝕜 _ _ x y


/-- A continuous linear endomorphism `T` of a Hilbert space is **positive** if it is self adjoint
  and `∀ x, 0 ≤ re ⟪T x, x⟫`. -/
def IsPositive (T : E →L[𝕜] E) : Prop :=
  IsSelfAdjoint T ∧ ∀ x, 0 ≤ T.reApplyInnerSelf x


theorem IsPositive.isSelfAdjoint {T : E →L[𝕜] E} (hT : IsPositive T) : IsSelfAdjoint T :=
  hT.1


theorem IsPositive.inner_nonneg_left {T : E →L[𝕜] E} (hT : IsPositive T) (x : E) :
    0 ≤ re ⟪T x, x⟫ :=
  hT.2 x


theorem IsPositive.inner_nonneg_right {T : E →L[𝕜] E} (hT : IsPositive T) (x : E) :
                          /-
                            𝕜 : Type u_1
                            E : Type u_2
                            inst✝³ : RCLike 𝕜
                            inst✝² : NormedAddCommGroup E
                            inst✝¹ : InnerProductSpace 𝕜 E
                            inst✝ : CompleteSpace E
                            T : ContinuousLinearMap (RingHom.id 𝕜) E E
                            hT : T.IsPositive
                            x : E
                            ⊢ LE.le 0 (RCLike.re (Inner.inner x (T x)))
                          -/
    0 ≤ re ⟪x, T x⟫ := by rw [inner_re_symm]; exact hT.inner_nonneg_left x
                                              /-
                                                🎉 no goals
                                              -/


theorem isPositive_zero : IsPositive (0 : E →L[𝕜] E) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    ⊢ ContinuousLinearMap.IsPositive 0
  -/
  refine ⟨.zero _, fun x => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    x : E
    ⊢ LE.le 0 (ContinuousLinearMap.reApplyInnerSelf 0 x)
  -/
  change 0 ≤ re ⟪_, _⟫
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    x : E
    ⊢ LE.le 0 (RCLike.re (Inner.inner (0 x) x))
  -/
  rw [zero_apply, inner_zero_left, ZeroHomClass.map_zero]
  /-
    🎉 no goals
  -/


theorem isPositive_one : IsPositive (1 : E →L[𝕜] E) :=
  ⟨.one _, fun _ => inner_self_nonneg⟩


theorem IsPositive.add {T S : E →L[𝕜] E} (hT : T.IsPositive) (hS : S.IsPositive) :
    (T + S).IsPositive := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T S : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    hS : S.IsPositive
    ⊢ (HAdd.hAdd T S).IsPositive
  -/
  refine ⟨hT.isSelfAdjoint.add hS.isSelfAdjoint, fun x => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T S : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    hS : S.IsPositive
    x : E
    ⊢ LE.le 0 ((HAdd.hAdd T S).reApplyInnerSelf x)
  -/
  rw [reApplyInnerSelf, add_apply, inner_add_left, map_add]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    T S : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    hS : S.IsPositive
    x : E
    ⊢ LE.le 0 (HAdd.hAdd (RCLike.re (Inner.inner (T x) x)) (RCLike.re (Inner.inner …
  -/
  exact add_nonneg (hT.inner_nonneg_left x) (hS.inner_nonneg_left x)
  /-
    🎉 no goals
  -/


theorem IsPositive.conj_adjoint {T : E →L[𝕜] E} (hT : T.IsPositive) (S : E →L[𝕜] F) :
    (S ∘L T ∘L S†).IsPositive := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    S : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ (S.comp (T.comp (ContinuousLinearMap.adjoint S))).IsPositive
  -/
  refine ⟨hT.isSelfAdjoint.conj_adjoint S, fun x => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    S : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : F
    ⊢ LE.le 0 ((S.comp (T.comp (ContinuousLinearMap.adjoint S))).reApplyInnerSelf x)
  -/
  rw [reApplyInnerSelf, comp_apply, ← adjoint_inner_right]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    S : ContinuousLinearMap (RingHom.id 𝕜) E F
    x : F
    ⊢ LE.le 0 (RCLike.re (Inner.inner ((T.comp (ContinuousLinearMap.adjoint S)) x) …
  -/
  exact hT.inner_nonneg_left _
  /-
    🎉 no goals
  -/


theorem IsPositive.adjoint_conj {T : E →L[𝕜] E} (hT : T.IsPositive) (S : F →L[𝕜] E) :
    (S† ∘L T ∘L S).IsPositive := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    S : ContinuousLinearMap (RingHom.id 𝕜) F E
    ⊢ ((ContinuousLinearMap.adjoint S).comp (T.comp S)).IsPositive
  -/
  convert hT.conj_adjoint (S†)
  /-
    case h.e'_7.h.e'_24.h.e'_24
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : InnerProductSpace 𝕜 E
    inst✝² : InnerProductSpace 𝕜 F
    inst✝¹ : CompleteSpace E
    inst✝ : CompleteSpace F
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    S : ContinuousLinearMap (RingHom.id 𝕜) F E
    ⊢ Eq S (ContinuousLinearMap.adjoint (ContinuousLinearMap.adjoint S))
  -/
  rw [adjoint_adjoint]
  /-
    🎉 no goals
  -/


theorem IsPositive.conj_orthogonalProjection (U : Submodule 𝕜 E) {T : E →L[𝕜] E} (hT : T.IsPositive)
    [CompleteSpace U] :
    (U.subtypeL ∘L
        orthogonalProjection U ∘L T ∘L U.subtypeL ∘L orthogonalProjection U).IsPositive := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    U : Submodule 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ (U.subtypeL.comp ((orthogonalProjection U).comp (T.comp (U.subtypeL.comp (or …
  -/
  have := hT.conj_adjoint (U.subtypeL ∘L orthogonalProjection U)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    U : Submodule 𝕜 E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    this : ((U.subtypeL.comp (orthogonalProjection U)).comp (T.comp (ContinuousLin …
    ⊢ (U.subtypeL.comp ((orthogonalProjection U).comp (T.comp (U.subtypeL.comp (or …
  -/
  rwa [(orthogonalProjection_isSelfAdjoint U).adjoint_eq] at this
  /-
    🎉 no goals
  -/


theorem IsPositive.orthogonalProjection_comp {T : E →L[𝕜] E} (hT : T.IsPositive) (U : Submodule 𝕜 E)
    [CompleteSpace U] : (orthogonalProjection U ∘L T ∘L U.subtypeL).IsPositive := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    ⊢ ((orthogonalProjection U).comp (T.comp U.subtypeL)).IsPositive
  -/
  have := hT.conj_adjoint (orthogonalProjection U : E →L[𝕜] U)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : RCLike 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : InnerProductSpace 𝕜 E
    inst✝¹ : CompleteSpace E
    T : ContinuousLinearMap (RingHom.id 𝕜) E E
    hT : T.IsPositive
    U : Submodule 𝕜 E
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem U x)
    this : ((orthogonalProjection U).comp (T.comp (ContinuousLinearMap.adjoint (or …
    ⊢ ((orthogonalProjection U).comp (T.comp U.subtypeL)).IsPositive
  -/
  rwa [U.adjoint_orthogonalProjection] at this
  /-
    🎉 no goals
  -/


lemma antilipschitz_of_forall_le_inner_map {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace 𝕜 H] (f : H →L[𝕜] H) {c : ℝ≥0} (hc : 0 < c)
    (h : ∀ x, ‖x‖ ^ 2 * c ≤ ‖⟪f x, x⟫_𝕜‖) : AntilipschitzWith c⁻¹ f := by
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace 𝕜 H
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : H), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    ⊢ AntilipschitzWith (Inv.inv c) ⇑f
  -/
  refine f.antilipschitz_of_bound (K := c⁻¹) fun x ↦ ?_
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace 𝕜 H
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : H), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    x : H
    ⊢ LE.le (Norm.norm x) (HMul.hMul (↑(Inv.inv c)) (Norm.norm (f x)))
  -/
  rw [NNReal.coe_inv, inv_mul_eq_div, le_div_iff₀ (by exact_mod_cast hc)]
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace 𝕜 H
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : H), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    x : H
    ⊢ LE.le (HMul.hMul (Norm.norm x) ↑c) (Norm.norm (f x))
  -/
  simp_rw [sq, mul_assoc] at h
  /-
    𝕜 : Type u_1
    inst✝² : RCLike 𝕜
    H : Type u_4
    inst✝¹ : NormedAddCommGroup H
    inst✝ : InnerProductSpace 𝕜 H
    f : ContinuousLinearMap (RingHom.id 𝕜) H H
    c : NNReal
    hc : LT.lt 0 c
    x : H
    h : ∀ (x : H), LE.le (HMul.hMul (Norm.norm x) (HMul.hMul (Norm.norm x) ↑c)) (N …
    ⊢ LE.le (HMul.hMul (Norm.norm x) ↑c) (Norm.norm (f x))
  -/
  by_cases hx0 : x = 0
    /-
      case pos
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : InnerProductSpace 𝕜 H
      f : ContinuousLinearMap (RingHom.id 𝕜) H H
      c : NNReal
      hc : LT.lt 0 c
      x : H
      h : ∀ (x : H), LE.le (HMul.hMul (Norm.norm x) (HMul.hMul (Norm.norm x) ↑c)) (N …
      hx0 : Eq x 0
      ⊢ LE.le (HMul.hMul (Norm.norm x) ↑c) (Norm.norm (f x))
    -/
  · simp [hx0]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : InnerProductSpace 𝕜 H
      f : ContinuousLinearMap (RingHom.id 𝕜) H H
      c : NNReal
      hc : LT.lt 0 c
      x : H
      h : ∀ (x : H), LE.le (HMul.hMul (Norm.norm x) (HMul.hMul (Norm.norm x) ↑c)) (N …
      hx0 : Not (Eq x 0)
      ⊢ LE.le (HMul.hMul (Norm.norm x) ↑c) (Norm.norm (f x))
    -/
  · apply (map_le_map_iff <| OrderIso.mulLeft₀ ‖x‖ (norm_pos_iff.mpr hx0)).mp
    /-
      case neg
      𝕜 : Type u_1
      inst✝² : RCLike 𝕜
      H : Type u_4
      inst✝¹ : NormedAddCommGroup H
      inst✝ : InnerProductSpace 𝕜 H
      f : ContinuousLinearMap (RingHom.id 𝕜) H H
      c : NNReal
      hc : LT.lt 0 c
      x : H
      h : ∀ (x : H), LE.le (HMul.hMul (Norm.norm x) (HMul.hMul (Norm.norm x) ↑c)) (N …
      hx0 : Not (Eq x 0)
      ⊢ LE.le ((OrderIso.mulLeft₀ (Norm.norm x) ⋯) (HMul.hMul (Norm.norm x) ↑c)) ((O …
    -/
    exact (h x).trans <| (norm_inner_le_norm _ _).trans <| (mul_comm _ _).le
    /-
      🎉 no goals
    -/


lemma isUnit_of_forall_le_norm_inner_map (f : E →L[𝕜] E) {c : ℝ≥0} (hc : 0 < c)
    (h : ∀ x, ‖x‖ ^ 2 * c ≤ ‖⟪f x, x⟫_𝕜‖) : IsUnit f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    ⊢ IsUnit f
  -/
  rw [isUnit_iff_bijective, bijective_iff_dense_range_and_antilipschitz]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    ⊢ And (Eq (LinearMap.range f).topologicalClosure Top.top) (Exists fun c => Ant …
  -/
  have h_anti : AntilipschitzWith c⁻¹ f := antilipschitz_of_forall_le_inner_map f hc h
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    h_anti : AntilipschitzWith (Inv.inv c) ⇑f
    ⊢ And (Eq (LinearMap.range f).topologicalClosure Top.top) (Exists fun c => Ant …
  -/
  refine ⟨?_, ⟨_, h_anti⟩⟩
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    h_anti : AntilipschitzWith (Inv.inv c) ⇑f
    ⊢ Eq (LinearMap.range f).topologicalClosure Top.top
  -/
  have _inst := h_anti.completeSpace_range_clm
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    h_anti : AntilipschitzWith (Inv.inv c) ⇑f
    _inst : CompleteSpace (Subtype fun x => Membership.mem (LinearMap.range f) x)
    ⊢ Eq (LinearMap.range f).topologicalClosure Top.top
  -/
  rw [Submodule.topologicalClosure_eq_top_iff, Submodule.eq_bot_iff]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    h_anti : AntilipschitzWith (Inv.inv c) ⇑f
    _inst : CompleteSpace (Subtype fun x => Membership.mem (LinearMap.range f) x)
    ⊢ ∀ (x : E), Membership.mem (LinearMap.range f).orthogonal x → Eq x 0
  -/
  intro x hx
  have : ‖x‖ ^ 2 * c = 0 := le_antisymm (by simpa only [hx (f x) ⟨x, rfl⟩, norm_zero] using h x)
    (by positivity)
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    c : NNReal
    hc : LT.lt 0 c
    h : ∀ (x : E), LE.le (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) (Norm.norm (In …
    h_anti : AntilipschitzWith (Inv.inv c) ⇑f
    _inst : CompleteSpace (Subtype fun x => Membership.mem (LinearMap.range f) x)
    x : E
    hx : Membership.mem (LinearMap.range f).orthogonal x
    this : Eq (HMul.hMul (HPow.hPow (Norm.norm x) 2) ↑c) 0
    ⊢ Eq x 0
  -/
  aesop
  /-
    🎉 no goals
  -/


theorem isPositive_iff_complex (T : E' →L[ℂ] E') :
    IsPositive T ↔ ∀ x, (re ⟪T x, x⟫_ℂ : ℂ) = ⟪T x, x⟫_ℂ ∧ 0 ≤ re ⟪T x, x⟫_ℂ := by
  simp_rw [IsPositive, forall_and, isSelfAdjoint_iff_isSymmetric,
    LinearMap.isSymmetric_iff_inner_map_self_real, conj_eq_iff_re]
  /-
    E' : Type u_4
    inst✝² : NormedAddCommGroup E'
    inst✝¹ : InnerProductSpace Complex E'
    inst✝ : CompleteSpace E'
    T : ContinuousLinearMap (RingHom.id Complex) E' E'
    ⊢ Iff (And (∀ (v : E'), Eq (↑(RCLike.re (Inner.inner (↑T v) v))) (Inner.inner  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The (Loewner) partial order on continuous linear maps on a Hilbert space determined by
`f ≤ g` if and only if `g - f` is a positive linear map (in the sense of
`ContinuousLinearMap.IsPositive`). With this partial order, the continuous linear maps form a
`StarOrderedRing`. -/
instance instLoewnerPartialOrder : PartialOrder (E →L[𝕜] E) where
  le f g := (g - f).IsPositive
                  /-
                    𝕜 : Type u_1
                    E : Type u_2
                    F : Type u_3
                    inst✝⁶ : RCLike 𝕜
                    inst✝⁵ : NormedAddCommGroup E
                    inst✝⁴ : NormedAddCommGroup F
                    inst✝³ : InnerProductSpace 𝕜 E
                    inst✝² : InnerProductSpace 𝕜 F
                    inst✝¹ : CompleteSpace E
                    inst✝ : CompleteSpace F
                    x✝ : ContinuousLinearMap (RingHom.id 𝕜) E E
                    ⊢ LE.le x✝ x✝
                  -/
  le_refl _ := by simpa using isPositive_zero
                  /-
                    🎉 no goals
                  -/
                             /-
                               𝕜 : Type u_1
                               E : Type u_2
                               F : Type u_3
                               inst✝⁶ : RCLike 𝕜
                               inst✝⁵ : NormedAddCommGroup E
                               inst✝⁴ : NormedAddCommGroup F
                               inst✝³ : InnerProductSpace 𝕜 E
                               inst✝² : InnerProductSpace 𝕜 F
                               inst✝¹ : CompleteSpace E
                               inst✝ : CompleteSpace F
                               x✝² x✝¹ x✝ : ContinuousLinearMap (RingHom.id 𝕜) E E
                               h₁ : LE.le x✝² x✝¹
                               h₂ : LE.le x✝¹ x✝
                               ⊢ LE.le x✝² x✝
                             -/
  le_trans _ _ _ h₁ h₂ := by simpa using h₁.add h₂
                             /-
                               🎉 no goals
                             -/
  le_antisymm f₁ f₂ h₁ h₂ := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) E E
      h₁ : LE.le f₁ f₂
      h₂ : LE.le f₂ f₁
      ⊢ Eq f₁ f₂
    -/
    rw [← sub_eq_zero]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : InnerProductSpace 𝕜 E
      inst✝² : InnerProductSpace 𝕜 F
      inst✝¹ : CompleteSpace E
      inst✝ : CompleteSpace F
      f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) E E
      h₁ : LE.le f₁ f₂
      h₂ : LE.le f₂ f₁
      ⊢ Eq (HSub.hSub f₁ f₂) 0
    -/
    have h_isSymm := isSelfAdjoint_iff_isSymmetric.mp <| IsPositive.isSelfAdjoint h₂
    exact_mod_cast h_isSymm.inner_map_self_eq_zero.mp fun x ↦ by
      apply RCLike.ext
      · rw [map_zero]
        apply le_antisymm
        · rw [← neg_nonneg, ← map_neg, ← inner_neg_left]
          simpa using h₁.inner_nonneg_left _
        · exact h₂.inner_nonneg_left _
      · rw [coe_sub, LinearMap.sub_apply, coe_coe, coe_coe, map_zero, ← sub_apply,
          ← h_isSymm.coe_reApplyInnerSelf_apply (T := f₁ - f₂) x, RCLike.ofReal_im]


lemma le_def (f g : E →L[𝕜] E) : f ≤ g ↔ (g - f).IsPositive := Iff.rfl


lemma nonneg_iff_isPositive (f : E →L[𝕜] E) : 0 ≤ f ↔ f.IsPositive := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝³ : RCLike 𝕜
    inst✝² : NormedAddCommGroup E
    inst✝¹ : InnerProductSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    ⊢ Iff (LE.le 0 f) f.IsPositive
  -/
  simpa using le_def 0 f
  /-
    🎉 no goals
  -/


