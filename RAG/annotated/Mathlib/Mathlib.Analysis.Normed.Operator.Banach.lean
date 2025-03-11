/-- A (possibly nonlinear) right inverse to a continuous linear map, which doesn't have to be
linear itself but which satisfies a bound `‖inverse x‖ ≤ C * ‖x‖`. A surjective continuous linear
map doesn't always have a continuous linear right inverse, but it always has a nonlinear inverse
in this sense, by Banach's open mapping theorem. -/
structure NonlinearRightInverse where
  toFun : F → E
  nnnorm : ℝ≥0
  bound' : ∀ y, ‖toFun y‖ ≤ nnnorm * ‖y‖
  right_inv' : ∀ y, f (toFun y) = y


instance : CoeFun (NonlinearRightInverse f) fun _ => F → E :=
  ⟨fun fsymm => fsymm.toFun⟩


@[simp]
theorem NonlinearRightInverse.right_inv {f : E →SL[σ] F} (fsymm : NonlinearRightInverse f) (y : F) :
    f (fsymm y) = y :=
  fsymm.right_inv' y


theorem NonlinearRightInverse.bound {f : E →SL[σ] F} (fsymm : NonlinearRightInverse f) (y : F) :
    ‖fsymm y‖ ≤ fsymm.nnnorm * ‖y‖ :=
  fsymm.bound' y


/-- Given a continuous linear equivalence, the inverse is in particular an instance of
`ContinuousLinearMap.NonlinearRightInverse` (which turns out to be linear). -/
noncomputable def ContinuousLinearEquiv.toNonlinearRightInverse
    [RingHomInvPair σ' σ] (f : E ≃SL[σ] F) :
    ContinuousLinearMap.NonlinearRightInverse (f : E →SL[σ] F) where
  toFun := f.invFun
  nnnorm := ‖(f.symm : F →SL[σ'] E)‖₊
  bound' _ := ContinuousLinearMap.le_opNorm (f.symm : F →SL[σ'] E) _
  right_inv' := f.apply_symm_apply


noncomputable instance [RingHomInvPair σ' σ] (f : E ≃SL[σ] F) :
    Inhabited (ContinuousLinearMap.NonlinearRightInverse (f : E →SL[σ] F)) :=
  ⟨f.toNonlinearRightInverse⟩


include σ' in
/-- First step of the proof of the Banach open mapping theorem (using completeness of `F`):
by Baire's theorem, there exists a ball in `E` whose image closure has nonempty interior.
Rescaling everything, it follows that any `y ∈ F` is arbitrarily well approached by
images of elements of norm at most `C * ‖y‖`.
For further use, we will only need such an element whose image
is within distance `‖y‖/2` of `y`, to apply an iterative process. -/
theorem exists_approx_preimage_norm_le (surj : Surjective f) :
    ∃ C ≥ 0, ∀ y, ∃ x, dist (f x) y ≤ 1 / 2 * ‖y‖ ∧ ‖x‖ ≤ C * ‖y‖ := by
  have A : ⋃ n : ℕ, closure (f '' ball 0 n) = Set.univ := by
    refine Subset.antisymm (subset_univ _) fun y _ => ?_
    rcases surj y with ⟨x, hx⟩
    rcases exists_nat_gt ‖x‖ with ⟨n, hn⟩
    refine mem_iUnion.2 ⟨n, subset_closure ?_⟩
    refine (mem_image _ _ _).2 ⟨x, ⟨?_, hx⟩⟩
    rwa [mem_ball, dist_eq_norm, sub_zero]
  have : ∃ (n : ℕ) (x : _), x ∈ interior (closure (f '' ball 0 n)) :=
    nonempty_interior_of_iUnion_of_closed (fun n => isClosed_closure) A
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomIsometric σ
    inst✝¹ : RingHomIsometric σ'
    inst✝ : CompleteSpace F
    surj : Function.Surjective ⇑f
    A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
    this : Exists fun n => Exists fun x => Membership.mem (interior (closure (Set. …
    ⊢ Exists fun C => And (GE.ge C 0) (∀ (y : F), Exists fun x => And (LE.le (Dist …
  -/
  simp only [mem_interior_iff_mem_nhds, Metric.mem_nhds_iff] at this
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomIsometric σ
    inst✝¹ : RingHomIsometric σ'
    inst✝ : CompleteSpace F
    surj : Function.Surjective ⇑f
    A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
    this : Exists fun n => Exists fun x => Exists fun ε => And (GT.gt ε 0) (HasSub …
    ⊢ Exists fun C => And (GE.ge C 0) (∀ (y : F), Exists fun x => And (LE.le (Dist …
  -/
  rcases this with ⟨n, a, ε, ⟨εpos, H⟩⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomIsometric σ
    inst✝¹ : RingHomIsometric σ'
    inst✝ : CompleteSpace F
    surj : Function.Surjective ⇑f
    A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
    n : Nat
    a : F
    ε : Real
    εpos : GT.gt ε 0
    H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
    ⊢ Exists fun C => And (GE.ge C 0) (∀ (y : F), Exists fun x => And (LE.le (Dist …
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomIsometric σ
    inst✝¹ : RingHomIsometric σ'
    inst✝ : CompleteSpace F
    surj : Function.Surjective ⇑f
    A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
    n : Nat
    a : F
    ε : Real
    εpos : GT.gt ε 0
    H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ⊢ Exists fun C => And (GE.ge C 0) (∀ (y : F), Exists fun x => And (LE.le (Dist …
  -/
  refine ⟨(ε / 2)⁻¹ * ‖c‖ * 2 * n, by positivity, fun y => ?_⟩
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝⁹ : NontriviallyNormedField 𝕜
    inst✝⁸ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝³ : RingHomInvPair σ σ'
    inst✝² : RingHomIsometric σ
    inst✝¹ : RingHomIsometric σ'
    inst✝ : CompleteSpace F
    surj : Function.Surjective ⇑f
    A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
    n : Nat
    a : F
    ε : Real
    εpos : GT.gt ε 0
    H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    y : F
    ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
  -/
  rcases eq_or_ne y 0 with rfl | hy
    /-
      case intro.intro.intro.intro.intro.inl
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) 0) (HMul.hMul (1 / 2) (Norm.norm …
    -/
  · use 0
    /-
      case h
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ⊢ And (LE.le (Dist.dist (f 0) 0) (HMul.hMul (1 / 2) (Norm.norm 0))) (LE.le (No …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.inr
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
  · have hc' : 1 < ‖σ c‖ := by simp only [RingHomIsometric.is_iso, hc]
    /-
      case intro.intro.intro.intro.intro.inr
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rcases rescale_to_shell hc' (half_pos εpos) hy with ⟨d, hd, ydlt, -, dinv⟩
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    let δ := ‖d‖ * ‖y‖ / 4
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    have δpos : 0 < δ := div_pos (mul_pos (norm_pos_iff.2 hd) (norm_pos_iff.2 hy)) (by norm_num)
    have : a + d • y ∈ ball a ε := by
      simp [dist_eq_norm, lt_of_le_of_lt ydlt.le (half_lt_self εpos)]
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rcases Metric.mem_closure_iff.1 (H this) _ δpos with ⟨z₁, z₁im, h₁⟩
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) z₁) δ
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rcases (mem_image _ _ _).1 z₁im with ⟨x₁, hx₁, xz₁⟩
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) z₁) δ
      x₁ : E
      hx₁ : Membership.mem (Metric.ball 0 ↑n) x₁
      xz₁ : Eq (f x₁) z₁
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rw [← xz₁] at h₁
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : Membership.mem (Metric.ball 0 ↑n) x₁
      xz₁ : Eq (f x₁) z₁
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rw [mem_ball, dist_eq_norm, sub_zero] at hx₁
    have : a ∈ ball a ε := by
      simp only [mem_ball, dist_self]
      exact εpos
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this✝ : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : LT.lt (Norm.norm x₁) ↑n
      xz₁ : Eq (f x₁) z₁
      this : Membership.mem (Metric.ball a ε) a
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rcases Metric.mem_closure_iff.1 (H this) _ δpos with ⟨z₂, z₂im, h₂⟩
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this✝ : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : LT.lt (Norm.norm x₁) ↑n
      xz₁ : Eq (f x₁) z₁
      this : Membership.mem (Metric.ball a ε) a
      z₂ : F
      z₂im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₂
      h₂ : LT.lt (Dist.dist a z₂) δ
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rcases (mem_image _ _ _).1 z₂im with ⟨x₂, hx₂, xz₂⟩
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this✝ : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : LT.lt (Norm.norm x₁) ↑n
      xz₁ : Eq (f x₁) z₁
      this : Membership.mem (Metric.ball a ε) a
      z₂ : F
      z₂im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₂
      h₂ : LT.lt (Dist.dist a z₂) δ
      x₂ : E
      hx₂ : Membership.mem (Metric.ball 0 ↑n) x₂
      xz₂ : Eq (f x₂) z₂
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rw [← xz₂] at h₂
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this✝ : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : LT.lt (Norm.norm x₁) ↑n
      xz₁ : Eq (f x₁) z₁
      this : Membership.mem (Metric.ball a ε) a
      z₂ : F
      z₂im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₂
      x₂ : E
      h₂ : LT.lt (Dist.dist a (f x₂)) δ
      hx₂ : Membership.mem (Metric.ball 0 ↑n) x₂
      xz₂ : Eq (f x₂) z₂
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rw [mem_ball, dist_eq_norm, sub_zero] at hx₂
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this✝ : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : LT.lt (Norm.norm x₁) ↑n
      xz₁ : Eq (f x₁) z₁
      this : Membership.mem (Metric.ball a ε) a
      z₂ : F
      z₂im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₂
      x₂ : E
      h₂ : LT.lt (Dist.dist a (f x₂)) δ
      hx₂ : LT.lt (Norm.norm x₂) ↑n
      xz₂ : Eq (f x₂) z₂
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    let x := x₁ - x₂
    have I : ‖f x - d • y‖ ≤ 2 * δ :=
      calc
        ‖f x - d • y‖ = ‖f x₁ - (a + d • y) - (f x₂ - a)‖ := by
          congr 1
          simp only [x, f.map_sub]
          abel
        _ ≤ ‖f x₁ - (a + d • y)‖ + ‖f x₂ - a‖ := norm_sub_le _ _
        _ ≤ δ + δ := by rw [dist_eq_norm'] at h₁ h₂; gcongr
        _ = 2 * δ := (two_mul _).symm
    have J : ‖f (σ' d⁻¹ • x) - y‖ ≤ 1 / 2 * ‖y‖ :=
      calc
        ‖f (σ' d⁻¹ • x) - y‖ = ‖d⁻¹ • f x - (d⁻¹ * d) • y‖ := by
          rwa [f.map_smulₛₗ _, inv_mul_cancel₀, one_smul, map_inv₀, map_inv₀,
            RingHomCompTriple.comp_apply, RingHom.id_apply]
        _ = ‖d⁻¹ • (f x - d • y)‖ := by rw [mul_smul, smul_sub]
        _ = ‖d‖⁻¹ * ‖f x - d • y‖ := by rw [norm_smul, norm_inv]
        _ ≤ ‖d‖⁻¹ * (2 * δ) := by gcongr
        _ = ‖d‖⁻¹ * ‖d‖ * ‖y‖ / 2 := by
          simp only [δ]
          ring
        _ = ‖y‖ / 2 := by
          rw [inv_mul_cancel₀, one_mul]
          simp [norm_eq_zero, hd]
        _ = 1 / 2 * ‖y‖ := by ring
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this✝ : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : LT.lt (Norm.norm x₁) ↑n
      xz₁ : Eq (f x₁) z₁
      this : Membership.mem (Metric.ball a ε) a
      z₂ : F
      z₂im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₂
      x₂ : E
      h₂ : LT.lt (Dist.dist a (f x₂)) δ
      hx₂ : LT.lt (Norm.norm x₂) ↑n
      xz₂ : Eq (f x₂) z₂
      x : E := HSub.hSub x₁ x₂
      I : LE.le (Norm.norm (HSub.hSub (f x) (HSMul.hSMul d y))) (HMul.hMul 2 δ)
      J : LE.le (Norm.norm (HSub.hSub (f (HSMul.hSMul (σ' (Inv.inv d)) x)) y)) (HMul …
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    rw [← dist_eq_norm] at J
    have K : ‖σ' d⁻¹ • x‖ ≤ (ε / 2)⁻¹ * ‖c‖ * 2 * ↑n * ‖y‖ :=
      calc
        ‖σ' d⁻¹ • x‖ = ‖d‖⁻¹ * ‖x₁ - x₂‖ := by rw [norm_smul, RingHomIsometric.is_iso, norm_inv]
        _ ≤ (ε / 2)⁻¹ * ‖c‖ * ‖y‖ * (n + n) := by
          gcongr
          · simpa using dinv
          · exact le_trans (norm_sub_le _ _) (by gcongr)
        _ = (ε / 2)⁻¹ * ‖c‖ * 2 * ↑n * ‖y‖ := by ring
    /-
      case intro.intro.intro.intro.intro.inr.intro.intro.intro.intro.intro.intro.int …
      𝕜 : Type u_1
      𝕜' : Type u_2
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : NontriviallyNormedField 𝕜'
      σ : RingHom 𝕜 𝕜'
      E : Type u_3
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      F : Type u_4
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜' F
      f : ContinuousLinearMap σ E F
      σ' : RingHom 𝕜' 𝕜
      inst✝³ : RingHomInvPair σ σ'
      inst✝² : RingHomIsometric σ
      inst✝¹ : RingHomIsometric σ'
      inst✝ : CompleteSpace F
      surj : Function.Surjective ⇑f
      A : Eq (Set.iUnion fun n => closure (Set.image (⇑f) (Metric.ball 0 ↑n))) Set.u …
      n : Nat
      a : F
      ε : Real
      εpos : GT.gt ε 0
      H : HasSubset.Subset (Metric.ball a ε) (closure (Set.image (⇑f) (Metric.ball 0 …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      y : F
      hy : Ne y 0
      hc' : LT.lt 1 (Norm.norm (σ c))
      d : 𝕜'
      hd : Ne d 0
      ydlt : LT.lt (Norm.norm (HSMul.hSMul d y)) (HDiv.hDiv ε 2)
      dinv : LE.le (Inv.inv (Norm.norm d)) (HMul.hMul (HMul.hMul (Inv.inv (HDiv.hDiv …
      δ : Real := HDiv.hDiv (HMul.hMul (Norm.norm d) (Norm.norm y)) 4
      δpos : LT.lt 0 δ
      this✝ : Membership.mem (Metric.ball a ε) (HAdd.hAdd a (HSMul.hSMul d y))
      z₁ : F
      z₁im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₁
      x₁ : E
      h₁ : LT.lt (Dist.dist (HAdd.hAdd a (HSMul.hSMul d y)) (f x₁)) δ
      hx₁ : LT.lt (Norm.norm x₁) ↑n
      xz₁ : Eq (f x₁) z₁
      this : Membership.mem (Metric.ball a ε) a
      z₂ : F
      z₂im : Membership.mem (Set.image (⇑f) (Metric.ball 0 ↑n)) z₂
      x₂ : E
      h₂ : LT.lt (Dist.dist a (f x₂)) δ
      hx₂ : LT.lt (Norm.norm x₂) ↑n
      xz₂ : Eq (f x₂) z₂
      x : E := HSub.hSub x₁ x₂
      I : LE.le (Norm.norm (HSub.hSub (f x) (HSMul.hSMul d y))) (HMul.hMul 2 δ)
      J : LE.le (Dist.dist (f (HSMul.hSMul (σ' (Inv.inv d)) x)) y) (HMul.hMul (1 / 2 …
      K : LE.le (Norm.norm (HSMul.hSMul (σ' (Inv.inv d)) x)) (HMul.hMul (HMul.hMul ( …
      ⊢ Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / 2) (Norm.norm …
    -/
    exact ⟨σ' d⁻¹ • x, J, K⟩
    /-
      🎉 no goals
    -/


/-- The Banach open mapping theorem: if a bounded linear map between Banach spaces is onto, then
any point has a preimage with controlled norm. -/
theorem exists_preimage_norm_le (surj : Surjective f) :
    ∃ C > 0, ∀ y, ∃ x, f x = y ∧ ‖x‖ ≤ C * ‖y‖ := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    ⊢ Exists fun C => And (GT.gt C 0) (∀ (y : F), Exists fun x => And (Eq (f x) y) …
  -/
  obtain ⟨C, C0, hC⟩ := exists_approx_preimage_norm_le f surj
  /- Second step of the proof: starting from `y`, we want an exact preimage of `y`. Let `g y` be
    the approximate preimage of `y` given by the first step, and `h y = y - f(g y)` the part that
    has no preimage yet. We will iterate this process, taking the approximate preimage of `h y`,
    leaving only `h^2 y` without preimage yet, and so on. Let `u n` be the approximate preimage
    of `h^n y`. Then `u` is a converging series, and by design the sum of the series is a
    preimage of `y`. This uses completeness of `E`. -/
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    hC : ∀ (y : F), Exists fun x => And (LE.le (Dist.dist (f x) y) (HMul.hMul (1 / …
    ⊢ Exists fun C => And (GT.gt C 0) (∀ (y : F), Exists fun x => And (Eq (f x) y) …
  -/
  choose g hg using hC
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    ⊢ Exists fun C => And (GT.gt C 0) (∀ (y : F), Exists fun x => And (Eq (f x) y) …
  -/
  let h y := y - f (g y)
  have hle : ∀ y, ‖h y‖ ≤ 1 / 2 * ‖y‖ := by
    intro y
    rw [← dist_eq_norm, dist_comm]
    exact (hg y).1
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    ⊢ Exists fun C => And (GT.gt C 0) (∀ (y : F), Exists fun x => And (Eq (f x) y) …
  -/
  refine ⟨2 * C + 1, by linarith, fun y => ?_⟩
  have hnle : ∀ n : ℕ, ‖h^[n] y‖ ≤ (1 / 2) ^ n * ‖y‖ := by
    intro n
    induction n with
    | zero => simp only [one_div, one_mul, iterate_zero_apply, pow_zero, le_rfl]
    | succ n IH =>
      rw [iterate_succ']
      apply le_trans (hle _) _
      rw [pow_succ', mul_assoc]
      gcongr
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  let u n := g (h^[n] y)
  have ule : ∀ n, ‖u n‖ ≤ (1 / 2) ^ n * (C * ‖y‖) := fun n ↦ by
    apply le_trans (hg _).2
    calc
      C * ‖h^[n] y‖ ≤ C * ((1 / 2) ^ n * ‖y‖) := mul_le_mul_of_nonneg_left (hnle n) C0
      _ = (1 / 2) ^ n * (C * ‖y‖) := by ring
  have sNu : Summable fun n => ‖u n‖ := by
    refine .of_nonneg_of_le (fun n => norm_nonneg _) ule ?_
    exact Summable.mul_right _ (summable_geometric_of_lt_one (by norm_num) (by norm_num))
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    u : Nat → E := fun n => g (Nat.iterate h n y)
    ule : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul (HPow.hPow (1 / 2) n) (H …
    sNu : Summable fun n => Norm.norm (u n)
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  have su : Summable u := sNu.of_norm
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    u : Nat → E := fun n => g (Nat.iterate h n y)
    ule : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul (HPow.hPow (1 / 2) n) (H …
    sNu : Summable fun n => Norm.norm (u n)
    su : Summable u
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  let x := tsum u
  have x_ineq : ‖x‖ ≤ (2 * C + 1) * ‖y‖ :=
    calc
      ‖x‖ ≤ ∑' n, ‖u n‖ := norm_tsum_le_tsum_norm sNu
      _ ≤ ∑' n, (1 / 2) ^ n * (C * ‖y‖) :=
        tsum_le_tsum ule sNu (Summable.mul_right _ summable_geometric_two)
      _ = (∑' n, (1 / 2) ^ n) * (C * ‖y‖) := tsum_mul_right
      _ = 2 * C * ‖y‖ := by rw [tsum_geometric_two, mul_assoc]
      _ ≤ 2 * C * ‖y‖ + ‖y‖ := le_add_of_nonneg_right (norm_nonneg y)
      _ = (2 * C + 1) * ‖y‖ := by ring
  have fsumeq : ∀ n : ℕ, f (∑ i ∈ Finset.range n, u i) = y - h^[n] y := by
    intro n
    induction n with
    | zero => simp [f.map_zero]
    | succ n IH => rw [sum_range_succ, f.map_add, IH, iterate_succ_apply', sub_add]
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    u : Nat → E := fun n => g (Nat.iterate h n y)
    ule : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul (HPow.hPow (1 / 2) n) (H …
    sNu : Summable fun n => Norm.norm (u n)
    su : Summable u
    x : E := tsum u
    x_ineq : LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd (HMul.hMul 2 C) 1) (Norm.no …
    fsumeq : ∀ (n : Nat), Eq (f ((Finset.range n).sum fun i => u i)) (HSub.hSub y  …
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  have : Tendsto (fun n => ∑ i ∈ Finset.range n, u i) atTop (𝓝 x) := su.hasSum.tendsto_sum_nat
  have L₁ : Tendsto (fun n => f (∑ i ∈ Finset.range n, u i)) atTop (𝓝 (f x)) :=
    (f.continuous.tendsto _).comp this
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    u : Nat → E := fun n => g (Nat.iterate h n y)
    ule : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul (HPow.hPow (1 / 2) n) (H …
    sNu : Summable fun n => Norm.norm (u n)
    su : Summable u
    x : E := tsum u
    x_ineq : LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd (HMul.hMul 2 C) 1) (Norm.no …
    fsumeq : ∀ (n : Nat), Eq (f ((Finset.range n).sum fun i => u i)) (HSub.hSub y  …
    this : Filter.Tendsto (fun n => (Finset.range n).sum fun i => u i) Filter.atTo …
    L₁ : Filter.Tendsto (fun n => f ((Finset.range n).sum fun i => u i)) Filter.at …
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  simp only [fsumeq] at L₁
  have L₂ : Tendsto (fun n => y - h^[n] y) atTop (𝓝 (y - 0)) := by
    refine tendsto_const_nhds.sub ?_
    rw [tendsto_iff_norm_sub_tendsto_zero]
    simp only [sub_zero]
    refine squeeze_zero (fun _ => norm_nonneg _) hnle ?_
    rw [← zero_mul ‖y‖]
    refine (_root_.tendsto_pow_atTop_nhds_zero_of_lt_one ?_ ?_).mul tendsto_const_nhds <;> norm_num
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    u : Nat → E := fun n => g (Nat.iterate h n y)
    ule : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul (HPow.hPow (1 / 2) n) (H …
    sNu : Summable fun n => Norm.norm (u n)
    su : Summable u
    x : E := tsum u
    x_ineq : LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd (HMul.hMul 2 C) 1) (Norm.no …
    fsumeq : ∀ (n : Nat), Eq (f ((Finset.range n).sum fun i => u i)) (HSub.hSub y  …
    this : Filter.Tendsto (fun n => (Finset.range n).sum fun i => u i) Filter.atTo …
    L₁ : Filter.Tendsto (fun n => HSub.hSub y (Nat.iterate h n y)) Filter.atTop (n …
    L₂ : Filter.Tendsto (fun n => HSub.hSub y (Nat.iterate h n y)) Filter.atTop (n …
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  have feq : f x = y - 0 := tendsto_nhds_unique L₁ L₂
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    u : Nat → E := fun n => g (Nat.iterate h n y)
    ule : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul (HPow.hPow (1 / 2) n) (H …
    sNu : Summable fun n => Norm.norm (u n)
    su : Summable u
    x : E := tsum u
    x_ineq : LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd (HMul.hMul 2 C) 1) (Norm.no …
    fsumeq : ∀ (n : Nat), Eq (f ((Finset.range n).sum fun i => u i)) (HSub.hSub y  …
    this : Filter.Tendsto (fun n => (Finset.range n).sum fun i => u i) Filter.atTo …
    L₁ : Filter.Tendsto (fun n => HSub.hSub y (Nat.iterate h n y)) Filter.atTop (n …
    L₂ : Filter.Tendsto (fun n => HSub.hSub y (Nat.iterate h n y)) Filter.atTop (n …
    feq : Eq (f x) (HSub.hSub y 0)
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  rw [sub_zero] at feq
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    C : Real
    C0 : GE.ge C 0
    g : F → E
    hg : ∀ (y : F), And (LE.le (Dist.dist (f (g y)) y) (HMul.hMul (1 / 2) (Norm.no …
    h : F → F := fun y => HSub.hSub y (f (g y))
    hle : ∀ (y : F), LE.le (Norm.norm (h y)) (HMul.hMul (1 / 2) (Norm.norm y))
    y : F
    hnle : ∀ (n : Nat), LE.le (Norm.norm (Nat.iterate h n y)) (HMul.hMul (HPow.hPo …
    u : Nat → E := fun n => g (Nat.iterate h n y)
    ule : ∀ (n : Nat), LE.le (Norm.norm (u n)) (HMul.hMul (HPow.hPow (1 / 2) n) (H …
    sNu : Summable fun n => Norm.norm (u n)
    su : Summable u
    x : E := tsum u
    x_ineq : LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd (HMul.hMul 2 C) 1) (Norm.no …
    fsumeq : ∀ (n : Nat), Eq (f ((Finset.range n).sum fun i => u i)) (HSub.hSub y  …
    this : Filter.Tendsto (fun n => (Finset.range n).sum fun i => u i) Filter.atTo …
    L₁ : Filter.Tendsto (fun n => HSub.hSub y (Nat.iterate h n y)) Filter.atTop (n …
    L₂ : Filter.Tendsto (fun n => HSub.hSub y (Nat.iterate h n y)) Filter.atTop (n …
    feq : Eq (f x) y
    ⊢ Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hMul (HAdd.hAdd  …
  -/
  exact ⟨x, feq, x_ineq⟩
  /-
    🎉 no goals
  -/


/-- The Banach open mapping theorem: a surjective bounded linear map between Banach spaces is
open. -/
protected theorem isOpenMap (surj : Surjective f) : IsOpenMap f := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    ⊢ IsOpenMap ⇑f
  -/
  intro s hs
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    ⊢ IsOpen (Set.image (⇑f) s)
  -/
  rcases exists_preimage_norm_le f surj with ⟨C, Cpos, hC⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    ⊢ IsOpen (Set.image (⇑f) s)
  -/
  refine isOpen_iff.2 fun y yfs => ?_
  /-
    case intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    y : F
    yfs : Membership.mem (Set.image (⇑f) s) y
    ⊢ Exists fun ε => And (GT.gt ε 0) (HasSubset.Subset (Metric.ball y ε) (Set.ima …
  -/
  rcases yfs with ⟨x, xs, fxy⟩
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    y : F
    x : E
    xs : Membership.mem s x
    fxy : Eq (f x) y
    ⊢ Exists fun ε => And (GT.gt ε 0) (HasSubset.Subset (Metric.ball y ε) (Set.ima …
  -/
  rcases isOpen_iff.1 hs x xs with ⟨ε, εpos, hε⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    y : F
    x : E
    xs : Membership.mem s x
    fxy : Eq (f x) y
    ε : Real
    εpos : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) s
    ⊢ Exists fun ε => And (GT.gt ε 0) (HasSubset.Subset (Metric.ball y ε) (Set.ima …
  -/
  refine ⟨ε / C, div_pos εpos Cpos, fun z hz => ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    y : F
    x : E
    xs : Membership.mem s x
    fxy : Eq (f x) y
    ε : Real
    εpos : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) s
    z : F
    hz : Membership.mem (Metric.ball y (HDiv.hDiv ε C)) z
    ⊢ Membership.mem (Set.image (⇑f) s) z
  -/
  rcases hC (z - y) with ⟨w, wim, wnorm⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    y : F
    x : E
    xs : Membership.mem s x
    fxy : Eq (f x) y
    ε : Real
    εpos : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) s
    z : F
    hz : Membership.mem (Metric.ball y (HDiv.hDiv ε C)) z
    w : E
    wim : Eq (f w) (HSub.hSub z y)
    wnorm : LE.le (Norm.norm w) (HMul.hMul C (Norm.norm (HSub.hSub z y)))
    ⊢ Membership.mem (Set.image (⇑f) s) z
  -/
  have : f (x + w) = z := by rw [f.map_add, wim, fxy, add_sub_cancel]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    y : F
    x : E
    xs : Membership.mem s x
    fxy : Eq (f x) y
    ε : Real
    εpos : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) s
    z : F
    hz : Membership.mem (Metric.ball y (HDiv.hDiv ε C)) z
    w : E
    wim : Eq (f w) (HSub.hSub z y)
    wnorm : LE.le (Norm.norm w) (HMul.hMul C (Norm.norm (HSub.hSub z y)))
    this : Eq (f (HAdd.hAdd x w)) z
    ⊢ Membership.mem (Set.image (⇑f) s) z
  -/
  rw [← this]
  have : x + w ∈ ball x ε :=
    calc
      dist (x + w) x = ‖w‖ := by
        rw [dist_eq_norm]
        simp
      _ ≤ C * ‖z - y‖ := wnorm
      _ < C * (ε / C) := by
        apply mul_lt_mul_of_pos_left _ Cpos
        rwa [mem_ball, dist_eq_norm] at hz
      _ = ε := mul_div_cancel₀ _ (ne_of_gt Cpos)

  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    f : ContinuousLinearMap σ E F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    surj : Function.Surjective ⇑f
    s : Set E
    hs : IsOpen s
    C : Real
    Cpos : GT.gt C 0
    hC : ∀ (y : F), Exists fun x => And (Eq (f x) y) (LE.le (Norm.norm x) (HMul.hM …
    y : F
    x : E
    xs : Membership.mem s x
    fxy : Eq (f x) y
    ε : Real
    εpos : GT.gt ε 0
    hε : HasSubset.Subset (Metric.ball x ε) s
    z : F
    hz : Membership.mem (Metric.ball y (HDiv.hDiv ε C)) z
    w : E
    wim : Eq (f w) (HSub.hSub z y)
    wnorm : LE.le (Norm.norm w) (HMul.hMul C (Norm.norm (HSub.hSub z y)))
    this✝ : Eq (f (HAdd.hAdd x w)) z
    this : Membership.mem (Metric.ball x ε) (HAdd.hAdd x w)
    ⊢ Membership.mem (Set.image (⇑f) s) (f (HAdd.hAdd x w))
  -/
  exact Set.mem_image_of_mem _ (hε this)
  /-
    🎉 no goals
  -/


theorem isQuotientMap (surj : Surjective f) : IsQuotientMap f :=
  (f.isOpenMap surj).isQuotientMap f.continuous surj


@[deprecated (since := "2024-10-22")]
alias quotientMap := isQuotientMap


theorem _root_.AffineMap.isOpenMap {F : Type*} [NormedAddCommGroup F] [NormedSpace 𝕜 F]
    [CompleteSpace F] {P Q : Type*} [MetricSpace P] [NormedAddTorsor E P] [MetricSpace Q]
    [NormedAddTorsor F Q] (f : P →ᵃ[𝕜] Q) (hf : Continuous f) (surj : Surjective f) :
    IsOpenMap f :=
  AffineMap.isOpenMap_linear_iff.mp <|
    ContinuousLinearMap.isOpenMap { f.linear with cont := AffineMap.continuous_linear_iff.mpr hf }
      (f.linear_surjective_iff.mpr surj)


theorem interior_preimage (hsurj : Surjective f) (s : Set F) :
    interior (f ⁻¹' s) = f ⁻¹' interior s :=
  ((f.isOpenMap hsurj).preimage_interior_eq_interior_preimage f.continuous s).symm


theorem closure_preimage (hsurj : Surjective f) (s : Set F) : closure (f ⁻¹' s) = f ⁻¹' closure s :=
  ((f.isOpenMap hsurj).preimage_closure_eq_closure_preimage f.continuous s).symm


theorem frontier_preimage (hsurj : Surjective f) (s : Set F) :
    frontier (f ⁻¹' s) = f ⁻¹' frontier s :=
  ((f.isOpenMap hsurj).preimage_frontier_eq_frontier_preimage f.continuous s).symm


theorem exists_nonlinearRightInverse_of_surjective (f : E →SL[σ] F)
    (hsurj : LinearMap.range f = ⊤) :
    ∃ fsymm : NonlinearRightInverse f, 0 < fsymm.nnnorm := by
  choose C hC fsymm h using
    exists_preimage_norm_le _ (LinearMap.range_eq_top.1 hsurj)
  use {
      toFun := fsymm
      nnnorm := ⟨C, hC.lt.le⟩
      bound' := fun y => (h y).2
      right_inv' := fun y => (h y).1 }
  /-
    case h
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap σ E F
    hsurj : Eq (LinearMap.range f) Top.top
    C : Real
    hC : GT.gt C 0
    fsymm : F → E
    h : ∀ (y : F), And (Eq (f (fsymm y)) y) (LE.le (Norm.norm (fsymm y)) (HMul.hMu …
    ⊢ LT.lt 0 { toFun := fsymm, nnnorm := ⟨C, ⋯⟩, bound' := ⋯, right_inv' := ⋯ }.n …
  -/
  exact hC
  /-
    🎉 no goals
  -/


/-- A surjective continuous linear map between Banach spaces admits a (possibly nonlinear)
controlled right inverse. In general, it is not possible to ensure that such a right inverse
is linear (take for instance the map from `E` to `E/F` where `F` is a closed subspace of `E`
without a closed complement. Then it doesn't have a continuous linear right inverse.) -/
noncomputable irreducible_def nonlinearRightInverseOfSurjective (f : E →SL[σ] F)
  (hsurj : LinearMap.range f = ⊤) : NonlinearRightInverse f :=
  Classical.choose (exists_nonlinearRightInverse_of_surjective f hsurj)


theorem nonlinearRightInverseOfSurjective_nnnorm_pos (f : E →SL[σ] F)
    (hsurj : LinearMap.range f = ⊤) : 0 < (nonlinearRightInverseOfSurjective f hsurj).nnnorm := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap σ E F
    hsurj : Eq (LinearMap.range f) Top.top
    ⊢ LT.lt 0 (f.nonlinearRightInverseOfSurjective hsurj).nnnorm
  -/
  rw [nonlinearRightInverseOfSurjective]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁸ : NormedAddCommGroup E
    inst✝⁷ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁴ : RingHomInvPair σ σ'
    inst✝³ : RingHomIsometric σ
    inst✝² : RingHomIsometric σ'
    inst✝¹ : CompleteSpace F
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap σ E F
    hsurj : Eq (LinearMap.range f) Top.top
    ⊢ LT.lt 0 (Classical.choose ⋯).nnnorm
  -/
  exact Classical.choose_spec (exists_nonlinearRightInverse_of_surjective f hsurj)
  /-
    🎉 no goals
  -/


/-- If a bounded linear map is a bijection, then its inverse is also a bounded linear map. -/
@[continuity]
theorem continuous_symm (e : E ≃ₛₗ[σ] F) (h : Continuous e) : Continuous e.symm := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁵ : RingHomInvPair σ σ'
    inst✝⁴ : RingHomIsometric σ
    inst✝³ : RingHomIsometric σ'
    inst✝² : CompleteSpace F
    inst✝¹ : CompleteSpace E
    inst✝ : RingHomInvPair σ' σ
    e : LinearEquiv σ E F
    h : Continuous ⇑e
    ⊢ Continuous ⇑e.symm
  -/
  rw [continuous_def]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁵ : RingHomInvPair σ σ'
    inst✝⁴ : RingHomIsometric σ
    inst✝³ : RingHomIsometric σ'
    inst✝² : CompleteSpace F
    inst✝¹ : CompleteSpace E
    inst✝ : RingHomInvPair σ' σ
    e : LinearEquiv σ E F
    h : Continuous ⇑e
    ⊢ ∀ (s : Set E), IsOpen s → IsOpen (Set.preimage (⇑e.symm) s)
  -/
  intro s hs
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁵ : RingHomInvPair σ σ'
    inst✝⁴ : RingHomIsometric σ
    inst✝³ : RingHomIsometric σ'
    inst✝² : CompleteSpace F
    inst✝¹ : CompleteSpace E
    inst✝ : RingHomInvPair σ' σ
    e : LinearEquiv σ E F
    h : Continuous ⇑e
    s : Set E
    hs : IsOpen s
    ⊢ IsOpen (Set.preimage (⇑e.symm) s)
  -/
  rw [← e.image_eq_preimage]
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁵ : RingHomInvPair σ σ'
    inst✝⁴ : RingHomIsometric σ
    inst✝³ : RingHomIsometric σ'
    inst✝² : CompleteSpace F
    inst✝¹ : CompleteSpace E
    inst✝ : RingHomInvPair σ' σ
    e : LinearEquiv σ E F
    h : Continuous ⇑e
    s : Set E
    hs : IsOpen s
    ⊢ IsOpen (Set.image (⇑e) s)
  -/
  rw [← e.coe_coe] at h ⊢
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁵ : RingHomInvPair σ σ'
    inst✝⁴ : RingHomIsometric σ
    inst✝³ : RingHomIsometric σ'
    inst✝² : CompleteSpace F
    inst✝¹ : CompleteSpace E
    inst✝ : RingHomInvPair σ' σ
    e : LinearEquiv σ E F
    h : Continuous ⇑↑e
    s : Set E
    hs : IsOpen s
    ⊢ IsOpen (Set.image (⇑↑e) s)
  -/
  exact ContinuousLinearMap.isOpenMap (σ := σ) ⟨_, h⟩ e.surjective s hs
  /-
    🎉 no goals
  -/


/-- Associating to a linear equivalence between Banach spaces a continuous linear equivalence when
the direct map is continuous, thanks to the Banach open mapping theorem that ensures that the
inverse map is also continuous. -/
def toContinuousLinearEquivOfContinuous (e : E ≃ₛₗ[σ] F) (h : Continuous e) : E ≃SL[σ] F :=
  { e with
    continuous_toFun := h
    continuous_invFun := e.continuous_symm h }


@[simp]
theorem coeFn_toContinuousLinearEquivOfContinuous (e : E ≃ₛₗ[σ] F) (h : Continuous e) :
    ⇑(e.toContinuousLinearEquivOfContinuous h) = e :=
  rfl


@[simp]
theorem coeFn_toContinuousLinearEquivOfContinuous_symm (e : E ≃ₛₗ[σ] F) (h : Continuous e) :
    ⇑(e.toContinuousLinearEquivOfContinuous h).symm = e.symm :=
  rfl


/-- An injective continuous linear map with a closed range defines a continuous linear equivalence
between its domain and its range. -/
noncomputable def equivRange (f : E →SL[σ] F) (hinj : Injective f) (hclo : IsClosed (range f)) :
    E ≃SL[σ] LinearMap.range f :=
  have : CompleteSpace (LinearMap.range f) := hclo.completeSpace_coe
  LinearEquiv.toContinuousLinearEquivOfContinuous (LinearEquiv.ofInjective f.toLinearMap hinj) <|
    (f.continuous.codRestrict fun x ↦ LinearMap.mem_range_self f x).congr fun _ ↦ rfl


@[simp]
theorem coe_linearMap_equivRange (f : E →SL[σ] F) (hinj : Injective f) (hclo : IsClosed (range f)) :
    f.equivRange hinj hclo = f.rangeRestrict :=
  rfl


@[simp]
theorem coe_equivRange (f : E →SL[σ] F) (hinj : Injective f) (hclo : IsClosed (range f)) :
    (f.equivRange hinj hclo : E → LinearMap.range f) = f.rangeRestrict :=
  rfl


/-- Convert a bijective continuous linear map `f : E →SL[σ] F` from a Banach space to a normed space
to a continuous linear equivalence. -/
noncomputable def ofBijective (f : E →SL[σ] F) (hinj : ker f = ⊥) (hsurj : LinearMap.range f = ⊤) :
    E ≃SL[σ] F :=
  (LinearEquiv.ofBijective ↑f
        ⟨LinearMap.ker_eq_bot.mp hinj,
          LinearMap.range_eq_top.mp hsurj⟩).toContinuousLinearEquivOfContinuous
    -- Porting note: added `by convert`
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          inst✝¹¹ : NontriviallyNormedField 𝕜
          inst✝¹⁰ : NontriviallyNormedField 𝕜'
          σ : RingHom 𝕜 𝕜'
          E : Type u_3
          inst✝⁹ : NormedAddCommGroup E
          inst✝⁸ : NormedSpace 𝕜 E
          F : Type u_4
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedSpace 𝕜' F
          f✝ : ContinuousLinearMap σ E F
          σ' : RingHom 𝕜' 𝕜
          inst✝⁵ : RingHomInvPair σ σ'
          inst✝⁴ : RingHomIsometric σ
          inst✝³ : RingHomIsometric σ'
          inst✝² : CompleteSpace F
          inst✝¹ : CompleteSpace E
          inst✝ : RingHomInvPair σ' σ
          f : ContinuousLinearMap σ E F
          hinj : Eq (LinearMap.ker f) Bot.bot
          hsurj : Eq (LinearMap.range f) Top.top
          ⊢ Continuous ⇑(LinearEquiv.ofBijective ↑f ⋯)
        -/
    (by convert f.continuous)
        /-
          🎉 no goals
        -/


@[simp]
theorem coeFn_ofBijective (f : E →SL[σ] F) (hinj : ker f = ⊥) (hsurj : LinearMap.range f = ⊤) :
    ⇑(ofBijective f hinj hsurj) = f :=
  rfl


theorem coe_ofBijective (f : E →SL[σ] F) (hinj : ker f = ⊥) (hsurj : LinearMap.range f = ⊤) :
    ↑(ofBijective f hinj hsurj) = f := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁵ : RingHomInvPair σ σ'
    inst✝⁴ : RingHomIsometric σ
    inst✝³ : RingHomIsometric σ'
    inst✝² : CompleteSpace F
    inst✝¹ : CompleteSpace E
    inst✝ : RingHomInvPair σ' σ
    f : ContinuousLinearMap σ E F
    hinj : Eq (LinearMap.ker f) Bot.bot
    hsurj : Eq (LinearMap.range f) Top.top
    ⊢ Eq (↑(ContinuousLinearEquiv.ofBijective f hinj hsurj)) f
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    σ : RingHom 𝕜 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    F : Type u_4
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : NormedSpace 𝕜' F
    σ' : RingHom 𝕜' 𝕜
    inst✝⁵ : RingHomInvPair σ σ'
    inst✝⁴ : RingHomIsometric σ
    inst✝³ : RingHomIsometric σ'
    inst✝² : CompleteSpace F
    inst✝¹ : CompleteSpace E
    inst✝ : RingHomInvPair σ' σ
    f : ContinuousLinearMap σ E F
    hinj : Eq (LinearMap.ker f) Bot.bot
    hsurj : Eq (LinearMap.range f) Top.top
    x✝ : E
    ⊢ Eq (↑(ContinuousLinearEquiv.ofBijective f hinj hsurj) x✝) (f x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem ofBijective_symm_apply_apply (f : E →SL[σ] F) (hinj : ker f = ⊥)
    (hsurj : LinearMap.range f = ⊤) (x : E) : (ofBijective f hinj hsurj).symm (f x) = x :=
  (ofBijective f hinj hsurj).symm_apply_apply x


@[simp]
theorem ofBijective_apply_symm_apply (f : E →SL[σ] F) (hinj : ker f = ⊥)
    (hsurj : LinearMap.range f = ⊤) (y : F) : f ((ofBijective f hinj hsurj).symm y) = y :=
  (ofBijective f hinj hsurj).apply_symm_apply y


lemma _root_.ContinuousLinearMap.isUnit_iff_bijective {f : E →L[𝕜] E} :
    IsUnit f ↔ Bijective f := by
  /-
    𝕜 : Type u_1
    inst✝³ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : ContinuousLinearMap (RingHom.id 𝕜) E E
    ⊢ Iff (IsUnit f) (Function.Bijective ⇑f)
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : ContinuousLinearMap (RingHom.id 𝕜) E E
      ⊢ IsUnit f → Function.Bijective ⇑f
    -/
  · rintro ⟨f, rfl⟩
    /-
      case mp.intro
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : Units (ContinuousLinearMap (RingHom.id 𝕜) E E)
      ⊢ Function.Bijective ⇑↑f
    -/
    exact ofUnit f |>.bijective
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : ContinuousLinearMap (RingHom.id 𝕜) E E
      ⊢ Function.Bijective ⇑f → IsUnit f
    -/
  · refine fun h ↦ ⟨toUnit <| .ofBijective f ?_ ?_, rfl⟩ <;>
    /-
      case mpr.refine_1
      𝕜 : Type u_1
      inst✝³ : NontriviallyNormedField 𝕜
      E : Type u_3
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : ContinuousLinearMap (RingHom.id 𝕜) E E
      h : Function.Bijective ⇑f
      ⊢ Eq (LinearMap.ker f) Bot.bot
    -/
    /-
      🎉 no goals
    -/
    simp only [LinearMap.range_eq_top, LinearMapClass.ker_eq_bot, h.1, h.2]
    /-
      🎉 no goals
    -/


/-- Intermediate definition used to show
`ContinuousLinearMap.closed_complemented_range_of_isCompl_of_ker_eq_bot`.

This is `f.coprod G.subtypeL` as a `ContinuousLinearEquiv`. -/
noncomputable def coprodSubtypeLEquivOfIsCompl {F : Type*} [NormedAddCommGroup F] [NormedSpace 𝕜 F]
    [CompleteSpace F] (f : E →L[𝕜] F) {G : Submodule 𝕜 F}
    (h : IsCompl (LinearMap.range f) G) [CompleteSpace G] (hker : ker f = ⊥) : (E × G) ≃L[𝕜] F :=
  ContinuousLinearEquiv.ofBijective (f.coprod G.subtypeL)
    (by
      /-
        𝕜 : Type u_1
        𝕜' : Type u_2
        inst✝¹⁴ : NontriviallyNormedField 𝕜
        inst✝¹³ : NontriviallyNormedField 𝕜'
        σ : RingHom 𝕜 𝕜'
        E : Type u_3
        inst✝¹² : NormedAddCommGroup E
        inst✝¹¹ : NormedSpace 𝕜 E
        F✝ : Type u_4
        inst✝¹⁰ : NormedAddCommGroup F✝
        inst✝⁹ : NormedSpace 𝕜' F✝
        f✝ : ContinuousLinearMap σ E F✝
        σ' : RingHom 𝕜' 𝕜
        inst✝⁸ : RingHomInvPair σ σ'
        inst✝⁷ : RingHomIsometric σ
        inst✝⁶ : RingHomIsometric σ'
        inst✝⁵ : CompleteSpace F✝
        inst✝⁴ : CompleteSpace E
        F : Type u_5
        inst✝³ : NormedAddCommGroup F
        inst✝² : NormedSpace 𝕜 F
        inst✝¹ : CompleteSpace F
        f : ContinuousLinearMap (RingHom.id 𝕜) E F
        G : Submodule 𝕜 F
        h : IsCompl (LinearMap.range f) G
        inst✝ : CompleteSpace (Subtype fun x => Membership.mem G x)
        hker : Eq (LinearMap.ker f) Bot.bot
        ⊢ Eq (LinearMap.ker (f.coprod G.subtypeL)) Bot.bot
      -/
      rw [ker_coprod_of_disjoint_range]
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          inst✝¹⁴ : NontriviallyNormedField 𝕜
          inst✝¹³ : NontriviallyNormedField 𝕜'
          σ : RingHom 𝕜 𝕜'
          E : Type u_3
          inst✝¹² : NormedAddCommGroup E
          inst✝¹¹ : NormedSpace 𝕜 E
          F✝ : Type u_4
          inst✝¹⁰ : NormedAddCommGroup F✝
          inst✝⁹ : NormedSpace 𝕜' F✝
          f✝ : ContinuousLinearMap σ E F✝
          σ' : RingHom 𝕜' 𝕜
          inst✝⁸ : RingHomInvPair σ σ'
          inst✝⁷ : RingHomIsometric σ
          inst✝⁶ : RingHomIsometric σ'
          inst✝⁵ : CompleteSpace F✝
          inst✝⁴ : CompleteSpace E
          F : Type u_5
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          inst✝¹ : CompleteSpace F
          f : ContinuousLinearMap (RingHom.id 𝕜) E F
          G : Submodule 𝕜 F
          h : IsCompl (LinearMap.range f) G
          inst✝ : CompleteSpace (Subtype fun x => Membership.mem G x)
          hker : Eq (LinearMap.ker f) Bot.bot
          ⊢ Eq ((LinearMap.ker f).prod (LinearMap.ker G.subtypeL)) Bot.bot
        -/
      · rw [hker, Submodule.ker_subtypeL, Submodule.prod_bot]
        /-
          🎉 no goals
        -/
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          inst✝¹⁴ : NontriviallyNormedField 𝕜
          inst✝¹³ : NontriviallyNormedField 𝕜'
          σ : RingHom 𝕜 𝕜'
          E : Type u_3
          inst✝¹² : NormedAddCommGroup E
          inst✝¹¹ : NormedSpace 𝕜 E
          F✝ : Type u_4
          inst✝¹⁰ : NormedAddCommGroup F✝
          inst✝⁹ : NormedSpace 𝕜' F✝
          f✝ : ContinuousLinearMap σ E F✝
          σ' : RingHom 𝕜' 𝕜
          inst✝⁸ : RingHomInvPair σ σ'
          inst✝⁷ : RingHomIsometric σ
          inst✝⁶ : RingHomIsometric σ'
          inst✝⁵ : CompleteSpace F✝
          inst✝⁴ : CompleteSpace E
          F : Type u_5
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          inst✝¹ : CompleteSpace F
          f : ContinuousLinearMap (RingHom.id 𝕜) E F
          G : Submodule 𝕜 F
          h : IsCompl (LinearMap.range f) G
          inst✝ : CompleteSpace (Subtype fun x => Membership.mem G x)
          hker : Eq (LinearMap.ker f) Bot.bot
          ⊢ Disjoint (LinearMap.range f) (LinearMap.range G.subtypeL)
        -/
      · rw [Submodule.range_subtypeL]
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          inst✝¹⁴ : NontriviallyNormedField 𝕜
          inst✝¹³ : NontriviallyNormedField 𝕜'
          σ : RingHom 𝕜 𝕜'
          E : Type u_3
          inst✝¹² : NormedAddCommGroup E
          inst✝¹¹ : NormedSpace 𝕜 E
          F✝ : Type u_4
          inst✝¹⁰ : NormedAddCommGroup F✝
          inst✝⁹ : NormedSpace 𝕜' F✝
          f✝ : ContinuousLinearMap σ E F✝
          σ' : RingHom 𝕜' 𝕜
          inst✝⁸ : RingHomInvPair σ σ'
          inst✝⁷ : RingHomIsometric σ
          inst✝⁶ : RingHomIsometric σ'
          inst✝⁵ : CompleteSpace F✝
          inst✝⁴ : CompleteSpace E
          F : Type u_5
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          inst✝¹ : CompleteSpace F
          f : ContinuousLinearMap (RingHom.id 𝕜) E F
          G : Submodule 𝕜 F
          h : IsCompl (LinearMap.range f) G
          inst✝ : CompleteSpace (Subtype fun x => Membership.mem G x)
          hker : Eq (LinearMap.ker f) Bot.bot
          ⊢ Disjoint (LinearMap.range f) G
        -/
        exact h.disjoint)
        /-
          🎉 no goals
        -/
        /-
          𝕜 : Type u_1
          𝕜' : Type u_2
          inst✝¹⁴ : NontriviallyNormedField 𝕜
          inst✝¹³ : NontriviallyNormedField 𝕜'
          σ : RingHom 𝕜 𝕜'
          E : Type u_3
          inst✝¹² : NormedAddCommGroup E
          inst✝¹¹ : NormedSpace 𝕜 E
          F✝ : Type u_4
          inst✝¹⁰ : NormedAddCommGroup F✝
          inst✝⁹ : NormedSpace 𝕜' F✝
          f✝ : ContinuousLinearMap σ E F✝
          σ' : RingHom 𝕜' 𝕜
          inst✝⁸ : RingHomInvPair σ σ'
          inst✝⁷ : RingHomIsometric σ
          inst✝⁶ : RingHomIsometric σ'
          inst✝⁵ : CompleteSpace F✝
          inst✝⁴ : CompleteSpace E
          F : Type u_5
          inst✝³ : NormedAddCommGroup F
          inst✝² : NormedSpace 𝕜 F
          inst✝¹ : CompleteSpace F
          f : ContinuousLinearMap (RingHom.id 𝕜) E F
          G : Submodule 𝕜 F
          h : IsCompl (LinearMap.range f) G
          inst✝ : CompleteSpace (Subtype fun x => Membership.mem G x)
          hker : Eq (LinearMap.ker f) Bot.bot
          ⊢ Eq (LinearMap.range (f.coprod G.subtypeL)) Top.top
        -/
    (by simp only [range_coprod, Submodule.range_subtypeL, h.sup_eq_top])
        /-
          🎉 no goals
        -/


theorem range_eq_map_coprodSubtypeLEquivOfIsCompl {F : Type*} [NormedAddCommGroup F]
    [NormedSpace 𝕜 F] [CompleteSpace F] (f : E →L[𝕜] F) {G : Submodule 𝕜 F}
    (h : IsCompl (LinearMap.range f) G) [CompleteSpace G] (hker : ker f = ⊥) :
    LinearMap.range f =
      ((⊤ : Submodule 𝕜 E).prod (⊥ : Submodule 𝕜 G)).map
        (f.coprodSubtypeLEquivOfIsCompl h hker : E × G →ₗ[𝕜] F) := by
  rw [coprodSubtypeLEquivOfIsCompl, ContinuousLinearEquiv.coe_ofBijective,
    coe_coprod, LinearMap.coprod_map_prod, Submodule.map_bot, sup_bot_eq, Submodule.map_top]
  /-
    𝕜 : Type u_1
    inst✝⁷ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace 𝕜 E
    inst✝⁴ : CompleteSpace E
    F : Type u_5
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : CompleteSpace F
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    G : Submodule 𝕜 F
    h : IsCompl (LinearMap.range f) G
    inst✝ : CompleteSpace (Subtype fun x => Membership.mem G x)
    hker : Eq (LinearMap.ker f) Bot.bot
    ⊢ Eq (LinearMap.range f) (LinearMap.range ↑f)
  -/
  rfl
  /-
    🎉 no goals
  -/

/- TODO: remove the assumption `f.ker = ⊥` in the next lemma, by using the map induced by `f` on
`E / f.ker`, once we have quotient normed spaces. -/

theorem closed_complemented_range_of_isCompl_of_ker_eq_bot {F : Type*} [NormedAddCommGroup F]
    [NormedSpace 𝕜 F] [CompleteSpace F] (f : E →L[𝕜] F) (G : Submodule 𝕜 F)
    (h : IsCompl (LinearMap.range f) G) (hG : IsClosed (G : Set F)) (hker : ker f = ⊥) :
    IsClosed (LinearMap.range f : Set F) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    G : Submodule 𝕜 F
    h : IsCompl (LinearMap.range f) G
    hG : IsClosed ↑G
    hker : Eq (LinearMap.ker f) Bot.bot
    ⊢ IsClosed ↑(LinearMap.range f)
  -/
  haveI : CompleteSpace G := hG.completeSpace_coe
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    G : Submodule 𝕜 F
    h : IsCompl (LinearMap.range f) G
    hG : IsClosed ↑G
    hker : Eq (LinearMap.ker f) Bot.bot
    this : CompleteSpace (Subtype fun x => Membership.mem G x)
    ⊢ IsClosed ↑(LinearMap.range f)
  -/
  let g := coprodSubtypeLEquivOfIsCompl f h hker
  -- Porting note: was `rw [congr_arg coe ...]`
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    G : Submodule 𝕜 F
    h : IsCompl (LinearMap.range f) G
    hG : IsClosed ↑G
    hker : Eq (LinearMap.ker f) Bot.bot
    this : CompleteSpace (Subtype fun x => Membership.mem G x)
    g : ContinuousLinearEquiv (RingHom.id 𝕜) (Prod E (Subtype fun x => Membership. …
    ⊢ IsClosed ↑(LinearMap.range f)
  -/
  rw [range_eq_map_coprodSubtypeLEquivOfIsCompl f h hker]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    G : Submodule 𝕜 F
    h : IsCompl (LinearMap.range f) G
    hG : IsClosed ↑G
    hker : Eq (LinearMap.ker f) Bot.bot
    this : CompleteSpace (Subtype fun x => Membership.mem G x)
    g : ContinuousLinearEquiv (RingHom.id 𝕜) (Prod E (Subtype fun x => Membership. …
    ⊢ IsClosed ↑(Submodule.map (↑↑(f.coprodSubtypeLEquivOfIsCompl h hker)) (Top.to …
  -/
  apply g.toHomeomorph.isClosed_image.2
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    f : ContinuousLinearMap (RingHom.id 𝕜) E F
    G : Submodule 𝕜 F
    h : IsCompl (LinearMap.range f) G
    hG : IsClosed ↑G
    hker : Eq (LinearMap.ker f) Bot.bot
    this : CompleteSpace (Subtype fun x => Membership.mem G x)
    g : ContinuousLinearEquiv (RingHom.id 𝕜) (Prod E (Subtype fun x => Membership. …
    ⊢ IsClosed ↑(Top.top.prod Bot.bot)
  -/
  exact isClosed_univ.prod isClosed_singleton
  /-
    🎉 no goals
  -/


/-- The **closed graph theorem** : a linear map between two Banach spaces whose graph is closed
is continuous. -/
theorem LinearMap.continuous_of_isClosed_graph (hg : IsClosed (g.graph : Set <| E × F)) :
    Continuous g := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : IsClosed ↑g.graph
    ⊢ Continuous ⇑g
  -/
  letI : CompleteSpace g.graph := completeSpace_coe_iff_isComplete.mpr hg.isComplete
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : IsClosed ↑g.graph
    this : CompleteSpace (Subtype fun x => Membership.mem g.graph x) := completeSp …
    ⊢ Continuous ⇑g
  -/
  let φ₀ : E →ₗ[𝕜] E × F := LinearMap.id.prod g
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : IsClosed ↑g.graph
    this : CompleteSpace (Subtype fun x => Membership.mem g.graph x) := completeSp …
    φ₀ : LinearMap (RingHom.id 𝕜) E (Prod E F) := LinearMap.id.prod g
    ⊢ Continuous ⇑g
  -/
  have : Function.LeftInverse Prod.fst φ₀ := fun x => rfl
  let φ : E ≃ₗ[𝕜] g.graph :=
    (LinearEquiv.ofLeftInverse this).trans (LinearEquiv.ofEq _ _ g.graph_eq_range_prod.symm)
  let ψ : g.graph ≃L[𝕜] E :=
    φ.symm.toContinuousLinearEquivOfContinuous continuous_subtype_val.fst
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : IsClosed ↑g.graph
    this✝ : CompleteSpace (Subtype fun x => Membership.mem g.graph x) := completeS …
    φ₀ : LinearMap (RingHom.id 𝕜) E (Prod E F) := LinearMap.id.prod g
    this : Function.LeftInverse Prod.fst ⇑φ₀
    φ : LinearEquiv (RingHom.id 𝕜) E (Subtype fun x => Membership.mem g.graph x) : …
    ψ : ContinuousLinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem g.gr …
    ⊢ Continuous ⇑g
  -/
  exact (continuous_subtype_val.comp ψ.symm.continuous).snd
  /-
    🎉 no goals
  -/


/-- A useful form of the **closed graph theorem** : let `f` be a linear map between two Banach
spaces. To show that `f` is continuous, it suffices to show that for any convergent sequence
`uₙ ⟶ x`, if `f(uₙ) ⟶ y` then `y = f(x)`. -/
theorem LinearMap.continuous_of_seq_closed_graph
    (hg : ∀ (u : ℕ → E) (x y), Tendsto u atTop (𝓝 x) → Tendsto (g ∘ u) atTop (𝓝 y) → y = g x) :
    Continuous g := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : ∀ (u : Nat → E) (x : E) (y : F), Filter.Tendsto u Filter.atTop (nhds x) → …
    ⊢ Continuous ⇑g
  -/
  refine g.continuous_of_isClosed_graph (IsSeqClosed.isClosed ?_)
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : ∀ (u : Nat → E) (x : E) (y : F), Filter.Tendsto u Filter.atTop (nhds x) → …
    ⊢ IsSeqClosed ↑g.graph
  -/
  rintro φ ⟨x, y⟩ hφg hφ
  /-
    case mk
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : ∀ (u : Nat → E) (x : E) (y : F), Filter.Tendsto u Filter.atTop (nhds x) → …
    φ : Nat → Prod E F
    x : E
    y : F
    hφg : ∀ (n : Nat), Membership.mem (↑g.graph) (φ n)
    hφ : Filter.Tendsto φ Filter.atTop (nhds { fst := x, snd := y })
    ⊢ Membership.mem ↑g.graph { fst := x, snd := y }
  -/
  refine hg (Prod.fst ∘ φ) x y ((continuous_fst.tendsto _).comp hφ) ?_
  have : g ∘ Prod.fst ∘ φ = Prod.snd ∘ φ := by
    ext n
    exact (hφg n).symm
  /-
    case mk
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : ∀ (u : Nat → E) (x : E) (y : F), Filter.Tendsto u Filter.atTop (nhds x) → …
    φ : Nat → Prod E F
    x : E
    y : F
    hφg : ∀ (n : Nat), Membership.mem (↑g.graph) (φ n)
    hφ : Filter.Tendsto φ Filter.atTop (nhds { fst := x, snd := y })
    this : Eq (Function.comp (⇑g) (Function.comp Prod.fst φ)) (Function.comp Prod. …
    ⊢ Filter.Tendsto (Function.comp (⇑g) (Function.comp Prod.fst φ)) Filter.atTop  …
  -/
  rw [this]
  /-
    case mk
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : ∀ (u : Nat → E) (x : E) (y : F), Filter.Tendsto u Filter.atTop (nhds x) → …
    φ : Nat → Prod E F
    x : E
    y : F
    hφg : ∀ (n : Nat), Membership.mem (↑g.graph) (φ n)
    hφ : Filter.Tendsto φ Filter.atTop (nhds { fst := x, snd := y })
    this : Eq (Function.comp (⇑g) (Function.comp Prod.fst φ)) (Function.comp Prod. …
    ⊢ Filter.Tendsto (Function.comp Prod.snd φ) Filter.atTop (nhds y)
  -/
  exact (continuous_snd.tendsto _).comp hφ
  /-
    🎉 no goals
  -/


/-- Upgrade a `LinearMap` to a `ContinuousLinearMap` using the **closed graph theorem**. -/
def ofIsClosedGraph (hg : IsClosed (g.graph : Set <| E × F)) : E →L[𝕜] F where
  toLinearMap := g
  cont := g.continuous_of_isClosed_graph hg


@[simp]
theorem coeFn_ofIsClosedGraph (hg : IsClosed (g.graph : Set <| E × F)) :
    ⇑(ContinuousLinearMap.ofIsClosedGraph hg) = g :=
  rfl


theorem coe_ofIsClosedGraph (hg : IsClosed (g.graph : Set <| E × F)) :
    ↑(ContinuousLinearMap.ofIsClosedGraph hg) = g := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : IsClosed ↑g.graph
    ⊢ Eq (↑(ContinuousLinearMap.ofIsClosedGraph hg)) g
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : IsClosed ↑g.graph
    x✝ : E
    ⊢ Eq (↑(ContinuousLinearMap.ofIsClosedGraph hg) x✝) (g x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Upgrade a `LinearMap` to a `ContinuousLinearMap` using a variation on the
**closed graph theorem**. -/
def ofSeqClosedGraph
    (hg : ∀ (u : ℕ → E) (x y), Tendsto u atTop (𝓝 x) → Tendsto (g ∘ u) atTop (𝓝 y) → y = g x) :
    E →L[𝕜] F where
  toLinearMap := g
  cont := g.continuous_of_seq_closed_graph hg


@[simp]
theorem coeFn_ofSeqClosedGraph
    (hg : ∀ (u : ℕ → E) (x y), Tendsto u atTop (𝓝 x) → Tendsto (g ∘ u) atTop (𝓝 y) → y = g x) :
    ⇑(ContinuousLinearMap.ofSeqClosedGraph hg) = g :=
  rfl


theorem coe_ofSeqClosedGraph
    (hg : ∀ (u : ℕ → E) (x y), Tendsto u atTop (𝓝 x) → Tendsto (g ∘ u) atTop (𝓝 y) → y = g x) :
    ↑(ContinuousLinearMap.ofSeqClosedGraph hg) = g := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : ∀ (u : Nat → E) (x : E) (y : F), Filter.Tendsto u Filter.atTop (nhds x) → …
    ⊢ Eq (↑(ContinuousLinearMap.ofSeqClosedGraph hg)) g
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_3
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : CompleteSpace E
    F : Type u_5
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : CompleteSpace F
    g : LinearMap (RingHom.id 𝕜) E F
    hg : ∀ (u : Nat → E) (x : E) (y : F), Filter.Tendsto u Filter.atTop (nhds x) → …
    x✝ : E
    ⊢ Eq (↑(ContinuousLinearMap.ofSeqClosedGraph hg) x✝) (g x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma closed_range_of_antilipschitz {f : E →SL[σ] F} {c : ℝ≥0} (hf : AntilipschitzWith c f) :
    (LinearMap.range f).topologicalClosure = LinearMap.range f :=
  SetLike.ext'_iff.mpr <| (hf.isClosed_range f.uniformContinuous).closure_eq


lemma _root_.AntilipschitzWith.completeSpace_range_clm {f : E →SL[σ] F} {c : ℝ≥0}
    (hf : AntilipschitzWith c f) : CompleteSpace (LinearMap.range f) :=
  IsClosed.completeSpace_coe <| hf.isClosed_range f.uniformContinuous


lemma bijective_iff_dense_range_and_antilipschitz (f : E →SL[σ] F) :
    Bijective f ↔ (LinearMap.range f).topologicalClosure = ⊤ ∧ ∃ c, AntilipschitzWith c f := by
  /-
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    σ : RingHom 𝕜 𝕜'
    σ' : RingHom 𝕜' 𝕜
    inst✝⁷ : RingHomInvPair σ σ'
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    inst✝⁴ : CompleteSpace E
    inst✝³ : CompleteSpace F
    inst✝² : RingHomInvPair σ' σ
    inst✝¹ : RingHomIsometric σ
    inst✝ : RingHomIsometric σ'
    f : ContinuousLinearMap σ E F
    ⊢ Iff (Function.Bijective ⇑f) (And (Eq (LinearMap.range f).topologicalClosure  …
  -/
  refine ⟨fun h ↦ ⟨?eq_top, ?anti⟩, fun ⟨hd, c, hf⟩ ↦ ⟨hf.injective, ?surj⟩⟩
  /-
    case eq_top
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    σ : RingHom 𝕜 𝕜'
    σ' : RingHom 𝕜' 𝕜
    inst✝⁷ : RingHomInvPair σ σ'
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    inst✝⁴ : CompleteSpace E
    inst✝³ : CompleteSpace F
    inst✝² : RingHomInvPair σ' σ
    inst✝¹ : RingHomIsometric σ
    inst✝ : RingHomIsometric σ'
    f : ContinuousLinearMap σ E F
    h : Function.Bijective ⇑f
    ⊢ Eq (LinearMap.range f).topologicalClosure Top.top
  -/
  case eq_top => simpa [SetLike.ext'_iff] using h.2.denseRange.closure_eq
  case anti =>
    refine ⟨_, ContinuousLinearEquiv.ofBijective f ?_ ?_ |>.antilipschitz⟩ <;>
    simp only [LinearMap.range_eq_top, LinearMapClass.ker_eq_bot, h.1, h.2]
  /-
    case surj
    𝕜 : Type u_1
    𝕜' : Type u_2
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : NontriviallyNormedField 𝕜'
    E : Type u_3
    inst✝⁹ : NormedAddCommGroup E
    inst✝⁸ : NormedSpace 𝕜 E
    σ : RingHom 𝕜 𝕜'
    σ' : RingHom 𝕜' 𝕜
    inst✝⁷ : RingHomInvPair σ σ'
    F : Type u_4
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜' F
    inst✝⁴ : CompleteSpace E
    inst✝³ : CompleteSpace F
    inst✝² : RingHomInvPair σ' σ
    inst✝¹ : RingHomIsometric σ
    inst✝ : RingHomIsometric σ'
    f : ContinuousLinearMap σ E F
    x✝ : And (Eq (LinearMap.range f).topologicalClosure Top.top) (Exists fun c =>  …
    hd : Eq (LinearMap.range f).topologicalClosure Top.top
    c : NNReal
    hf : AntilipschitzWith c ⇑f
    ⊢ Function.Surjective ⇑f
  -/
  case surj => rwa [← LinearMap.range_eq_top, ← closed_range_of_antilipschitz hf]
  /-
    🎉 no goals
  -/


