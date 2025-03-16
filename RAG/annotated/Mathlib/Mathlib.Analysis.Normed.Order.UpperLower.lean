@[to_additive IsUpperSet.thickening]
protected theorem IsUpperSet.thickening' (hs : IsUpperSet s) (ε : ℝ) :
    IsUpperSet (thickening ε s) := by
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsUpperSet s
    ε : Real
    ⊢ IsUpperSet (Metric.thickening ε s)
  -/
  rw [← ball_mul_one]
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsUpperSet s
    ε : Real
    ⊢ IsUpperSet (HMul.hMul (Metric.ball 1 ε) s)
  -/
  exact hs.mul_left
  /-
    🎉 no goals
  -/


@[to_additive IsLowerSet.thickening]
protected theorem IsLowerSet.thickening' (hs : IsLowerSet s) (ε : ℝ) :
    IsLowerSet (thickening ε s) := by
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsLowerSet s
    ε : Real
    ⊢ IsLowerSet (Metric.thickening ε s)
  -/
  rw [← ball_mul_one]
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsLowerSet s
    ε : Real
    ⊢ IsLowerSet (HMul.hMul (Metric.ball 1 ε) s)
  -/
  exact hs.mul_left
  /-
    🎉 no goals
  -/


@[to_additive IsUpperSet.cthickening]
protected theorem IsUpperSet.cthickening' (hs : IsUpperSet s) (ε : ℝ) :
    IsUpperSet (cthickening ε s) := by
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsUpperSet s
    ε : Real
    ⊢ IsUpperSet (Metric.cthickening ε s)
  -/
  rw [cthickening_eq_iInter_thickening'']
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsUpperSet s
    ε : Real
    ⊢ IsUpperSet (Set.iInter fun ε_1 => Set.iInter fun x => Metric.thickening ε_1 s)
  -/
  exact isUpperSet_iInter₂ fun δ _ => hs.thickening' _
  /-
    🎉 no goals
  -/


@[to_additive IsLowerSet.cthickening]
protected theorem IsLowerSet.cthickening' (hs : IsLowerSet s) (ε : ℝ) :
    IsLowerSet (cthickening ε s) := by
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsLowerSet s
    ε : Real
    ⊢ IsLowerSet (Metric.cthickening ε s)
  -/
  rw [cthickening_eq_iInter_thickening'']
  /-
    α : Type u_1
    inst✝ : NormedOrderedGroup α
    s : Set α
    hs : IsLowerSet s
    ε : Real
    ⊢ IsLowerSet (Set.iInter fun ε_1 => Set.iInter fun x => Metric.thickening ε_1 s)
  -/
  exact isLowerSet_iInter₂ fun δ _ => hs.thickening' _
  /-
    🎉 no goals
  -/


@[to_additive upperClosure_interior_subset] lemma upperClosure_interior_subset' (s : Set α) :
    (upperClosure (interior s) : Set α) ⊆ interior (upperClosure s) :=
  upperClosure_min (interior_mono subset_upperClosure) (upperClosure s).upper.interior


@[to_additive lowerClosure_interior_subset] lemma lowerClosure_interior_subset' (s : Set α) :
    (lowerClosure (interior s) : Set α) ⊆ interior (lowerClosure s) :=
  lowerClosure_min (interior_mono subset_lowerClosure) (lowerClosure s).lower.interior


theorem IsUpperSet.mem_interior_of_forall_lt (hs : IsUpperSet s) (hx : x ∈ closure s)
    (h : ∀ i, x i < y i) : y ∈ interior s := by
  /-
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    ⊢ Membership.mem (interior s) y
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ⊢ Membership.mem (interior s) y
  -/
  obtain ⟨ε, hε, hxy⟩ := Pi.exists_forall_pos_add_lt h
  /-
    case intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    ⊢ Membership.mem (interior s) y
  -/
  obtain ⟨z, hz, hxz⟩ := Metric.mem_closure_iff.1 hx _ hε
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : LT.lt (Dist.dist x z) ε
    ⊢ Membership.mem (interior s) y
  -/
  rw [dist_pi_lt_iff hε] at hxz
  have hyz : ∀ i, z i < y i := by
    refine fun i => (hxy _).trans_le' (sub_le_iff_le_add'.1 <| (le_abs_self _).trans ?_)
    rw [← Real.norm_eq_abs, ← dist_eq_norm']
    exact (hxz _).le
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz : ∀ (i : ι), LT.lt (z i) (y i)
    ⊢ Membership.mem (interior s) y
  -/
  obtain ⟨δ, hδ, hyz⟩ := Pi.exists_forall_pos_add_lt hyz
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (z i) (y i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (z i) δ) (y i)
    ⊢ Membership.mem (interior s) y
  -/
  refine mem_interior.2 ⟨ball y δ, ?_, isOpen_ball, mem_ball_self hδ⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (z i) (y i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (z i) δ) (y i)
    ⊢ HasSubset.Subset (Metric.ball y δ) s
  -/
  rintro w hw
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (z i) (y i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (z i) δ) (y i)
    w : ι → Real
    hw : Membership.mem (Metric.ball y δ) w
    ⊢ Membership.mem s w
  -/
  refine hs (fun i => ?_) hz
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (z i) (y i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (z i) δ) (y i)
    w : ι → Real
    hw : Membership.mem (Metric.ball y δ) w
    i : ι
    ⊢ LE.le (z i) (w i)
  -/
  simp_rw [ball_pi _ hδ, Real.ball_eq_Ioo] at hw
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (x i) (y i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (x i) ε) (y i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (z i) (y i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (z i) δ) (y i)
    w : ι → Real
    i : ι
    hw : Membership.mem (Set.univ.pi fun b => Set.Ioo (HSub.hSub (y b) δ) (HAdd.hA …
    ⊢ LE.le (z i) (w i)
  -/
  exact ((lt_sub_iff_add_lt.2 <| hyz _).trans (hw _ <| mem_univ _).1).le
  /-
    🎉 no goals
  -/


theorem IsLowerSet.mem_interior_of_forall_lt (hs : IsLowerSet s) (hx : x ∈ closure s)
    (h : ∀ i, y i < x i) : y ∈ interior s := by
  /-
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    ⊢ Membership.mem (interior s) y
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ⊢ Membership.mem (interior s) y
  -/
  obtain ⟨ε, hε, hxy⟩ := Pi.exists_forall_pos_add_lt h
  /-
    case intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    ⊢ Membership.mem (interior s) y
  -/
  obtain ⟨z, hz, hxz⟩ := Metric.mem_closure_iff.1 hx _ hε
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : LT.lt (Dist.dist x z) ε
    ⊢ Membership.mem (interior s) y
  -/
  rw [dist_pi_lt_iff hε] at hxz
  have hyz : ∀ i, y i < z i := by
    refine fun i =>
      (lt_sub_iff_add_lt.2 <| hxy _).trans_le (sub_le_comm.1 <| (le_abs_self _).trans ?_)
    rw [← Real.norm_eq_abs, ← dist_eq_norm]
    exact (hxz _).le
  /-
    case intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz : ∀ (i : ι), LT.lt (y i) (z i)
    ⊢ Membership.mem (interior s) y
  -/
  obtain ⟨δ, hδ, hyz⟩ := Pi.exists_forall_pos_add_lt hyz
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (y i) (z i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) δ) (z i)
    ⊢ Membership.mem (interior s) y
  -/
  refine mem_interior.2 ⟨ball y δ, ?_, isOpen_ball, mem_ball_self hδ⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (y i) (z i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) δ) (z i)
    ⊢ HasSubset.Subset (Metric.ball y δ) s
  -/
  rintro w hw
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (y i) (z i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) δ) (z i)
    w : ι → Real
    hw : Membership.mem (Metric.ball y δ) w
    ⊢ Membership.mem s w
  -/
  refine hs (fun i => ?_) hz
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (y i) (z i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) δ) (z i)
    w : ι → Real
    hw : Membership.mem (Metric.ball y δ) w
    i : ι
    ⊢ LE.le (w i) (z i)
  -/
  simp_rw [ball_pi _ hδ, Real.ball_eq_Ioo] at hw
  /-
    case intro.intro.intro.intro.intro.intro.intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    x y : ι → Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    h : ∀ (i : ι), LT.lt (y i) (x i)
    val✝ : Fintype ι
    ε : Real
    hε : LT.lt 0 ε
    hxy : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) ε) (x i)
    z : ι → Real
    hz : Membership.mem s z
    hxz : ∀ (b : ι), LT.lt (Dist.dist (x b) (z b)) ε
    hyz✝ : ∀ (i : ι), LT.lt (y i) (z i)
    δ : Real
    hδ : LT.lt 0 δ
    hyz : ∀ (i : ι), LT.lt (HAdd.hAdd (y i) δ) (z i)
    w : ι → Real
    i : ι
    hw : Membership.mem (Set.univ.pi fun b => Set.Ioo (HSub.hSub (y b) δ) (HAdd.hA …
    ⊢ LE.le (w i) (z i)
  -/
  exact ((hw _ <| mem_univ _).2.trans <| hyz _).le
  /-
    🎉 no goals
  -/


lemma dist_inf_sup_pi (x y : ι → ℝ) : dist (x ⊓ y) (x ⊔ y) = dist x y := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    x y : ι → Real
    ⊢ Eq (Dist.dist (Min.min x y) (Max.max x y)) (Dist.dist x y)
  -/
  refine congr_arg NNReal.toReal (Finset.sup_congr rfl fun i _ ↦ ?_)
  simp only [Real.nndist_eq', max_sub_min_eq_abs, Pi.inf_apply,
    Pi.sup_apply, Real.nnabs_of_nonneg, abs_nonneg, Real.toNNReal_abs]


lemma dist_mono_left_pi : MonotoneOn (dist · y) (Ici y) := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    y : ι → Real
    ⊢ MonotoneOn (fun x => Dist.dist x y) (Set.Ici y)
  -/
  refine fun y₁ hy₁ y₂ hy₂ hy ↦ NNReal.coe_le_coe.2 (Finset.sup_mono_fun fun i _ ↦ ?_)
  rw [Real.nndist_eq, Real.nnabs_of_nonneg (sub_nonneg_of_le (‹y ≤ _› i : y i ≤ y₁ i)),
    Real.nndist_eq, Real.nnabs_of_nonneg (sub_nonneg_of_le (‹y ≤ _› i : y i ≤ y₂ i))]
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    y y₁ : ι → Real
    hy₁ : Membership.mem (Set.Ici y) y₁
    y₂ : ι → Real
    hy₂ : Membership.mem (Set.Ici y) y₂
    hy : LE.le y₁ y₂
    i : ι
    x✝ : Membership.mem Finset.univ i
    ⊢ LE.le (HSub.hSub (y₁ i) (y i)).toNNReal (HSub.hSub (y₂ i) (y i)).toNNReal
  -/
  exact Real.toNNReal_mono (sub_le_sub_right (hy _) _)
  /-
    🎉 no goals
  -/


lemma dist_mono_right_pi : MonotoneOn (dist x) (Ici x) := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    x : ι → Real
    ⊢ MonotoneOn (Dist.dist x) (Set.Ici x)
  -/
  simpa only [dist_comm _ x] using dist_mono_left_pi (y := x)
  /-
    🎉 no goals
  -/


lemma dist_anti_left_pi : AntitoneOn (dist · y) (Iic y) := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    y : ι → Real
    ⊢ AntitoneOn (fun x => Dist.dist x y) (Set.Iic y)
  -/
  refine fun y₁ hy₁ y₂ hy₂ hy ↦ NNReal.coe_le_coe.2 (Finset.sup_mono_fun fun i _ ↦ ?_)
  rw [Real.nndist_eq', Real.nnabs_of_nonneg (sub_nonneg_of_le (‹_ ≤ y› i : y₂ i ≤ y i)),
    Real.nndist_eq', Real.nnabs_of_nonneg (sub_nonneg_of_le (‹_ ≤ y› i : y₁ i ≤ y i))]
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    y y₁ : ι → Real
    hy₁ : Membership.mem (Set.Iic y) y₁
    y₂ : ι → Real
    hy₂ : Membership.mem (Set.Iic y) y₂
    hy : LE.le y₁ y₂
    i : ι
    x✝ : Membership.mem Finset.univ i
    ⊢ LE.le (HSub.hSub (y i) (y₂ i)).toNNReal (HSub.hSub (y i) (y₁ i)).toNNReal
  -/
  exact Real.toNNReal_mono (sub_le_sub_left (hy _) _)
  /-
    🎉 no goals
  -/


lemma dist_anti_right_pi : AntitoneOn (dist x) (Iic x) := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    x : ι → Real
    ⊢ AntitoneOn (Dist.dist x) (Set.Iic x)
  -/
  simpa only [dist_comm] using dist_anti_left_pi (y := x)
  /-
    🎉 no goals
  -/


lemma dist_le_dist_of_le_pi (ha : a₂ ≤ a₁) (h₁ : a₁ ≤ b₁) (hb : b₁ ≤ b₂) :
    dist a₁ b₁ ≤ dist a₂ b₂ :=
  (dist_mono_right_pi h₁ (h₁.trans hb) hb).trans <|
    dist_anti_left_pi (ha.trans <| h₁.trans hb) (h₁.trans hb) ha


theorem IsUpperSet.exists_subset_ball (hs : IsUpperSet s) (hx : x ∈ closure s) (hδ : 0 < δ) :
    ∃ y, closedBall y (δ / 4) ⊆ closedBall x δ ∧ closedBall y (δ / 4) ⊆ interior s := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    ⊢ Exists fun y => And (HasSubset.Subset (Metric.closedBall y (HDiv.hDiv δ 4))  …
  -/
  refine ⟨x + const _ (3 / 4 * δ), closedBall_subset_closedBall' ?_, ?_⟩
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsUpperSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ LE.le (HAdd.hAdd (HDiv.hDiv δ 4) (Dist.dist (HAdd.hAdd x (Function.const ι ( …
    -/
  · rw [dist_self_add_left]
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsUpperSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ LE.le (HAdd.hAdd (HDiv.hDiv δ 4) (Norm.norm (Function.const ι (HMul.hMul (3  …
    -/
    refine (add_le_add_left (pi_norm_const_le <| 3 / 4 * δ) _).trans_eq ?_
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsUpperSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv δ 4) (Norm.norm (HMul.hMul (3 / 4) δ))) δ
    -/
    simp only [norm_mul, norm_div, Real.norm_eq_abs]
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsUpperSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv δ 4) (HMul.hMul (HDiv.hDiv (abs 3) (abs 4)) (abs δ) …
    -/
    simp only [gt_iff_lt, zero_lt_three, abs_of_pos, zero_lt_four, abs_of_pos hδ]
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsUpperSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv δ 4) (HMul.hMul (3 / 4) δ)) δ
    -/
    ring
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    ⊢ HasSubset.Subset (Metric.closedBall (HAdd.hAdd x (Function.const ι (HMul.hMu …
  -/
  obtain ⟨y, hy, hxy⟩ := Metric.mem_closure_iff.1 hx _ (div_pos hδ zero_lt_four)
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Dist.dist x y) (HDiv.hDiv δ 4)
    ⊢ HasSubset.Subset (Metric.closedBall (HAdd.hAdd x (Function.const ι (HMul.hMu …
  -/
  refine fun z hz => hs.mem_interior_of_forall_lt (subset_closure hy) fun i => ?_
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Dist.dist x y) (HDiv.hDiv δ 4)
    z : ι → Real
    hz : Membership.mem (Metric.closedBall (HAdd.hAdd x (Function.const ι (HMul.hM …
    i : ι
    ⊢ LT.lt (y i) (z i)
  -/
  rw [mem_closedBall, dist_eq_norm'] at hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Dist.dist x y) (HDiv.hDiv δ 4)
    z : ι → Real
    hz : LE.le (Norm.norm (HSub.hSub (HAdd.hAdd x (Function.const ι (HMul.hMul (3  …
    i : ι
    ⊢ LT.lt (y i) (z i)
  -/
  rw [dist_eq_norm] at hxy
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Norm.norm (HSub.hSub x y)) (HDiv.hDiv δ 4)
    z : ι → Real
    hz : LE.le (Norm.norm (HSub.hSub (HAdd.hAdd x (Function.const ι (HMul.hMul (3  …
    i : ι
    ⊢ LT.lt (y i) (z i)
  -/
  replace hxy := (norm_le_pi_norm _ i).trans hxy.le
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    hz : LE.le (Norm.norm (HSub.hSub (HAdd.hAdd x (Function.const ι (HMul.hMul (3  …
    i : ι
    hxy : LE.le (Norm.norm (HSub.hSub x y i)) (HDiv.hDiv δ 4)
    ⊢ LT.lt (y i) (z i)
  -/
  replace hz := (norm_le_pi_norm _ i).trans hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    i : ι
    hxy : LE.le (Norm.norm (HSub.hSub x y i)) (HDiv.hDiv δ 4)
    hz : LE.le (Norm.norm (HSub.hSub (HAdd.hAdd x (Function.const ι (HMul.hMul (3  …
    ⊢ LT.lt (y i) (z i)
  -/
  dsimp at hxy hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    i : ι
    hxy : LE.le (abs (HSub.hSub (x i) (y i))) (HDiv.hDiv δ 4)
    hz : LE.le (abs (HSub.hSub (HAdd.hAdd (x i) (HMul.hMul (3 / 4) δ)) (z i))) (HD …
    ⊢ LT.lt (y i) (z i)
  -/
  rw [abs_sub_le_iff] at hxy hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsUpperSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    i : ι
    hxy : And (LE.le (HSub.hSub (x i) (y i)) (HDiv.hDiv δ 4)) (LE.le (HSub.hSub (y …
    hz : And (LE.le (HSub.hSub (HAdd.hAdd (x i) (HMul.hMul (3 / 4) δ)) (z i)) (HDi …
    ⊢ LT.lt (y i) (z i)
  -/
  linarith
  /-
    🎉 no goals
  -/


theorem IsLowerSet.exists_subset_ball (hs : IsLowerSet s) (hx : x ∈ closure s) (hδ : 0 < δ) :
    ∃ y, closedBall y (δ / 4) ⊆ closedBall x δ ∧ closedBall y (δ / 4) ⊆ interior s := by
  /-
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    ⊢ Exists fun y => And (HasSubset.Subset (Metric.closedBall y (HDiv.hDiv δ 4))  …
  -/
  refine ⟨x - const _ (3 / 4 * δ), closedBall_subset_closedBall' ?_, ?_⟩
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsLowerSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ LE.le (HAdd.hAdd (HDiv.hDiv δ 4) (Dist.dist (HSub.hSub x (Function.const ι ( …
    -/
  · rw [dist_self_sub_left]
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsLowerSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ LE.le (HAdd.hAdd (HDiv.hDiv δ 4) (Norm.norm (Function.const ι (HMul.hMul (3  …
    -/
    refine (add_le_add_left (pi_norm_const_le <| 3 / 4 * δ) _).trans_eq ?_
    simp only [norm_mul, norm_div, Real.norm_eq_abs, gt_iff_lt, zero_lt_three, abs_of_pos,
      zero_lt_four, abs_of_pos hδ]
    /-
      case refine_1
      ι : Type u_2
      inst✝ : Fintype ι
      s : Set (ι → Real)
      x : ι → Real
      δ : Real
      hs : IsLowerSet s
      hx : Membership.mem (closure s) x
      hδ : LT.lt 0 δ
      ⊢ Eq (HAdd.hAdd (HDiv.hDiv δ 4) (HMul.hMul (3 / 4) δ)) δ
    -/
    ring
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    ⊢ HasSubset.Subset (Metric.closedBall (HSub.hSub x (Function.const ι (HMul.hMu …
  -/
  obtain ⟨y, hy, hxy⟩ := Metric.mem_closure_iff.1 hx _ (div_pos hδ zero_lt_four)
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Dist.dist x y) (HDiv.hDiv δ 4)
    ⊢ HasSubset.Subset (Metric.closedBall (HSub.hSub x (Function.const ι (HMul.hMu …
  -/
  refine fun z hz => hs.mem_interior_of_forall_lt (subset_closure hy) fun i => ?_
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Dist.dist x y) (HDiv.hDiv δ 4)
    z : ι → Real
    hz : Membership.mem (Metric.closedBall (HSub.hSub x (Function.const ι (HMul.hM …
    i : ι
    ⊢ LT.lt (z i) (y i)
  -/
  rw [mem_closedBall, dist_eq_norm'] at hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Dist.dist x y) (HDiv.hDiv δ 4)
    z : ι → Real
    hz : LE.le (Norm.norm (HSub.hSub (HSub.hSub x (Function.const ι (HMul.hMul (3  …
    i : ι
    ⊢ LT.lt (z i) (y i)
  -/
  rw [dist_eq_norm] at hxy
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    hxy : LT.lt (Norm.norm (HSub.hSub x y)) (HDiv.hDiv δ 4)
    z : ι → Real
    hz : LE.le (Norm.norm (HSub.hSub (HSub.hSub x (Function.const ι (HMul.hMul (3  …
    i : ι
    ⊢ LT.lt (z i) (y i)
  -/
  replace hxy := (norm_le_pi_norm _ i).trans hxy.le
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    hz : LE.le (Norm.norm (HSub.hSub (HSub.hSub x (Function.const ι (HMul.hMul (3  …
    i : ι
    hxy : LE.le (Norm.norm (HSub.hSub x y i)) (HDiv.hDiv δ 4)
    ⊢ LT.lt (z i) (y i)
  -/
  replace hz := (norm_le_pi_norm _ i).trans hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    i : ι
    hxy : LE.le (Norm.norm (HSub.hSub x y i)) (HDiv.hDiv δ 4)
    hz : LE.le (Norm.norm (HSub.hSub (HSub.hSub x (Function.const ι (HMul.hMul (3  …
    ⊢ LT.lt (z i) (y i)
  -/
  dsimp at hxy hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    i : ι
    hxy : LE.le (abs (HSub.hSub (x i) (y i))) (HDiv.hDiv δ 4)
    hz : LE.le (abs (HSub.hSub (HSub.hSub (x i) (HMul.hMul (3 / 4) δ)) (z i))) (HD …
    ⊢ LT.lt (z i) (y i)
  -/
  rw [abs_sub_le_iff] at hxy hz
  /-
    case refine_2.intro.intro
    ι : Type u_2
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : ι → Real
    δ : Real
    hs : IsLowerSet s
    hx : Membership.mem (closure s) x
    hδ : LT.lt 0 δ
    y : ι → Real
    hy : Membership.mem s y
    z : ι → Real
    i : ι
    hxy : And (LE.le (HSub.hSub (x i) (y i)) (HDiv.hDiv δ 4)) (LE.le (HSub.hSub (y …
    hz : And (LE.le (HSub.hSub (HSub.hSub (x i) (HMul.hMul (3 / 4) δ)) (z i)) (HDi …
    ⊢ LT.lt (z i) (y i)
  -/
  linarith
  /-
    🎉 no goals
  -/


protected lemma IsClosed.upperClosure_pi (hs : IsClosed s) (hs' : BddBelow s) :
    IsClosed (upperClosure s : Set (ι → ℝ)) := by
  /-
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddBelow s
    ⊢ IsClosed ↑(upperClosure s)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddBelow s
    val✝ : Fintype ι
    ⊢ IsClosed ↑(upperClosure s)
  -/
  refine IsSeqClosed.isClosed fun f x hf hx ↦ ?_
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddBelow s
    val✝ : Fintype ι
    f : Nat → ι → Real
    x : ι → Real
    hf : ∀ (n : Nat), Membership.mem (↑(upperClosure s)) (f n)
    hx : Filter.Tendsto f Filter.atTop (nhds x)
    ⊢ Membership.mem (↑(upperClosure s)) x
  -/
  choose g hg hgf using hf
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddBelow s
    val✝ : Fintype ι
    f : Nat → ι → Real
    x : ι → Real
    hx : Filter.Tendsto f Filter.atTop (nhds x)
    g : Nat → ι → Real
    hg : ∀ (n : Nat), Membership.mem s (g n)
    hgf : ∀ (n : Nat), LE.le (g n) (f n)
    ⊢ Membership.mem (↑(upperClosure s)) x
  -/
  obtain ⟨a, ha⟩ := hx.bddAbove_range
  obtain ⟨b, hb, φ, hφ, hbf⟩ := tendsto_subseq_of_bounded (hs'.isBounded_inter bddAbove_Iic) fun n ↦
    ⟨hg n, (hgf _).trans <| ha <| mem_range_self _⟩
  exact ⟨b, closure_minimal inter_subset_left hs hb,
    le_of_tendsto_of_tendsto' hbf (hx.comp hφ.tendsto_atTop) fun _ ↦ hgf _⟩


protected lemma IsClosed.lowerClosure_pi (hs : IsClosed s) (hs' : BddAbove s) :
    IsClosed (lowerClosure s : Set (ι → ℝ)) := by
  /-
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddAbove s
    ⊢ IsClosed ↑(lowerClosure s)
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddAbove s
    val✝ : Fintype ι
    ⊢ IsClosed ↑(lowerClosure s)
  -/
  refine IsSeqClosed.isClosed fun f x hf hx ↦ ?_
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddAbove s
    val✝ : Fintype ι
    f : Nat → ι → Real
    x : ι → Real
    hf : ∀ (n : Nat), Membership.mem (↑(lowerClosure s)) (f n)
    hx : Filter.Tendsto f Filter.atTop (nhds x)
    ⊢ Membership.mem (↑(lowerClosure s)) x
  -/
  choose g hg hfg using hf
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddAbove s
    val✝ : Fintype ι
    f : Nat → ι → Real
    x : ι → Real
    hx : Filter.Tendsto f Filter.atTop (nhds x)
    g : Nat → ι → Real
    hg : ∀ (n : Nat), Membership.mem s (g n)
    hfg : ∀ (n : Nat), LE.le (f n) (g n)
    ⊢ Membership.mem (↑(lowerClosure s)) x
  -/
  haveI : BoundedGENhdsClass ℝ := by infer_instance
  /-
    case intro
    ι : Type u_2
    inst✝ : Finite ι
    s : Set (ι → Real)
    hs : IsClosed s
    hs' : BddAbove s
    val✝ : Fintype ι
    f : Nat → ι → Real
    x : ι → Real
    hx : Filter.Tendsto f Filter.atTop (nhds x)
    g : Nat → ι → Real
    hg : ∀ (n : Nat), Membership.mem s (g n)
    hfg : ∀ (n : Nat), LE.le (f n) (g n)
    this : BoundedGENhdsClass Real
    ⊢ Membership.mem (↑(lowerClosure s)) x
  -/
  obtain ⟨a, ha⟩ := hx.bddBelow_range
  obtain ⟨b, hb, φ, hφ, hbf⟩ := tendsto_subseq_of_bounded (hs'.isBounded_inter bddBelow_Ici) fun n ↦
    ⟨hg n, (ha <| mem_range_self _).trans <| hfg _⟩
  exact ⟨b, closure_minimal inter_subset_left hs hb,
    le_of_tendsto_of_tendsto' (hx.comp hφ.tendsto_atTop) hbf fun _ ↦ hfg _⟩


protected lemma IsClopen.upperClosure_pi (hs : IsClopen s) (hs' : BddBelow s) :
    IsClopen (upperClosure s : Set (ι → ℝ)) := ⟨hs.1.upperClosure_pi hs', hs.2.upperClosure⟩


protected lemma IsClopen.lowerClosure_pi (hs : IsClopen s) (hs' : BddAbove s) :
    IsClopen (lowerClosure s : Set (ι → ℝ)) := ⟨hs.1.lowerClosure_pi hs', hs.2.lowerClosure⟩


lemma closure_upperClosure_comm_pi (hs : BddBelow s) :
    closure (upperClosure s : Set (ι → ℝ)) = upperClosure (closure s) :=
  (closure_minimal (upperClosure_anti subset_closure) <|
      isClosed_closure.upperClosure_pi hs.closure).antisymm <|
    upperClosure_min (closure_mono subset_upperClosure) (upperClosure s).upper.closure


lemma closure_lowerClosure_comm_pi (hs : BddAbove s) :
    closure (lowerClosure s : Set (ι → ℝ)) = lowerClosure (closure s) :=
  (closure_minimal (lowerClosure_mono subset_closure) <|
        isClosed_closure.lowerClosure_pi hs.closure).antisymm <|
    lowerClosure_min (closure_mono subset_lowerClosure) (lowerClosure s).lower.closure


