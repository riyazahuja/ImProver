/-- An auxiliary type for the proof of Urysohn's lemma: a pair of a closed set `C` and its
open neighborhood `U`, together with the assumption that `C` satisfies the property `P C`. The
latter assumption will make it possible to prove simultaneously both versions of Urysohn's lemma,
in normal spaces (with `P` always true) and in locally compact spaces (with `P = IsCompact`).
We put also in the structure the assumption that, for any such pair, one may find an intermediate
pair in between satisfying `P`, to avoid carrying it around in the argument. -/
structure CU {X : Type*} [TopologicalSpace X] (P : Set X → Prop) where
  /-- The inner set in the inductive construction towards Urysohn's lemma -/
  protected C : Set X
  /-- The outer set in the inductive construction towards Urysohn's lemma -/
  protected U : Set X
  protected P_C : P C
  protected closed_C : IsClosed C
  protected open_U : IsOpen U
  protected subset : C ⊆ U
  protected hP : ∀ {c u : Set X}, IsClosed c → P c → IsOpen u → c ⊆ u →
    ∃ v, IsOpen v ∧ c ⊆ v ∧ closure v ⊆ u ∧ P (closure v)


/-- By assumption, for each `c : CU P` there exists an open set `u`
such that `c.C ⊆ u` and `closure u ⊆ c.U`. `c.left` is the pair `(c.C, u)`. -/
@[simps C]
def left (c : CU P) : CU P where
  C := c.C
  U := (c.hP c.closed_C c.P_C c.open_U c.subset).choose
  closed_C := c.closed_C
  P_C := c.P_C
  open_U := (c.hP c.closed_C c.P_C c.open_U c.subset).choose_spec.1
  subset := (c.hP c.closed_C c.P_C c.open_U c.subset).choose_spec.2.1
  hP := c.hP


/-- By assumption, for each `c : CU P` there exists an open set `u`
such that `c.C ⊆ u` and `closure u ⊆ c.U`. `c.right` is the pair `(closure u, c.U)`. -/
@[simps U]
def right (c : CU P) : CU P where
  C := closure (c.hP c.closed_C c.P_C c.open_U c.subset).choose
  U := c.U
  closed_C := isClosed_closure
  P_C := (c.hP c.closed_C c.P_C c.open_U c.subset).choose_spec.2.2.2
  open_U := c.open_U
  subset := (c.hP c.closed_C c.P_C c.open_U c.subset).choose_spec.2.2.1
  hP := c.hP


theorem left_U_subset_right_C (c : CU P) : c.left.U ⊆ c.right.C :=
  subset_closure


theorem left_U_subset (c : CU P) : c.left.U ⊆ c.U :=
  Subset.trans c.left_U_subset_right_C c.right.subset


theorem subset_right_C (c : CU P) : c.C ⊆ c.right.C :=
  Subset.trans c.left.subset c.left_U_subset_right_C


/-- `n`-th approximation to a continuous function `f : X → ℝ` such that `f = 0` on `c.C` and `f = 1`
outside of `c.U`. -/
noncomputable def approx : ℕ → CU P → X → ℝ
  | 0, c, x => indicator c.Uᶜ 1 x
  | n + 1, c, x => midpoint ℝ (approx n c.left x) (approx n c.right x)


theorem approx_of_mem_C (c : CU P) (n : ℕ) {x : X} (hx : x ∈ c.C) : c.approx n x = 0 := by
  induction n generalizing c with
  | zero => exact indicator_of_not_mem (fun (hU : x ∈ c.Uᶜ) => hU <| c.subset hx) _
  | succ n ihn =>
    simp only [approx]
    rw [ihn, ihn, midpoint_self]
    exacts [c.subset_right_C hx, hx]


theorem approx_of_nmem_U (c : CU P) (n : ℕ) {x : X} (hx : x ∉ c.U) : c.approx n x = 1 := by
  induction n generalizing c with
  | zero =>
    rw [← mem_compl_iff] at hx
    exact indicator_of_mem hx _
  | succ n ihn =>
    simp only [approx]
    rw [ihn, ihn, midpoint_self]
    exacts [hx, fun hU => hx <| c.left_U_subset hU]


theorem approx_nonneg (c : CU P) (n : ℕ) (x : X) : 0 ≤ c.approx n x := by
  induction n generalizing c with
  | zero => exact indicator_nonneg (fun _ _ => zero_le_one) _
  | succ n ihn =>
    simp only [approx, midpoint_eq_smul_add, invOf_eq_inv]
    refine mul_nonneg (inv_nonneg.2 zero_le_two) (add_nonneg ?_ ?_) <;> apply ihn


theorem approx_le_one (c : CU P) (n : ℕ) (x : X) : c.approx n x ≤ 1 := by
  induction n generalizing c with
  | zero => exact indicator_apply_le' (fun _ => le_rfl) fun _ => zero_le_one
  | succ n ihn =>
    simp only [approx, midpoint_eq_smul_add, invOf_eq_inv, smul_eq_mul, ← div_eq_inv_mul]
    have := add_le_add (ihn (left c)) (ihn (right c))
    norm_num at this
    exact Iff.mpr (div_le_one zero_lt_two) this


theorem bddAbove_range_approx (c : CU P) (x : X) : BddAbove (range fun n => c.approx n x) :=
  ⟨1, fun _ ⟨n, hn⟩ => hn ▸ c.approx_le_one n x⟩


theorem approx_le_approx_of_U_sub_C {c₁ c₂ : CU P} (h : c₁.U ⊆ c₂.C) (n₁ n₂ : ℕ) (x : X) :
    c₂.approx n₂ x ≤ c₁.approx n₁ x := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c₁ c₂ : Urysohns.CU P
    h : HasSubset.Subset c₁.U c₂.C
    n₁ n₂ : Nat
    x : X
    ⊢ LE.le (Urysohns.CU.approx n₂ c₂ x) (Urysohns.CU.approx n₁ c₁ x)
  -/
  by_cases hx : x ∈ c₁.U
  · calc
      approx n₂ c₂ x = 0 := approx_of_mem_C _ _ (h hx)
      _ ≤ approx n₁ c₁ x := approx_nonneg _ _ _
  · calc
      approx n₂ c₂ x ≤ 1 := approx_le_one _ _ _
      _ = approx n₁ c₁ x := (approx_of_nmem_U _ _ hx).symm


theorem approx_mem_Icc_right_left (c : CU P) (n : ℕ) (x : X) :
    c.approx n x ∈ Icc (c.right.approx n x) (c.left.approx n x) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    n : Nat
    x : X
    ⊢ Membership.mem (Set.Icc (Urysohns.CU.approx n c.right x) (Urysohns.CU.approx …
  -/
  induction' n with n ihn generalizing c
  · exact ⟨le_rfl, indicator_le_indicator_of_subset (compl_subset_compl.2 c.left_U_subset)
      (fun _ => zero_le_one) _⟩
    /-
      case succ
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      x : X
      n : Nat
      ihn : ∀ (c : Urysohns.CU P), Membership.mem (Set.Icc (Urysohns.CU.approx n c.r …
      c : Urysohns.CU P
      ⊢ Membership.mem (Set.Icc (Urysohns.CU.approx (HAdd.hAdd n 1) c.right x) (Urys …
    -/
  · simp only [approx, mem_Icc]
    /-
      case succ
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      x : X
      n : Nat
      ihn : ∀ (c : Urysohns.CU P), Membership.mem (Set.Icc (Urysohns.CU.approx n c.r …
      c : Urysohns.CU P
      ⊢ And (LE.le (midpoint Real (Urysohns.CU.approx n c.right.left x) (Urysohns.CU …
    -/
    refine ⟨midpoint_le_midpoint ?_ (ihn _).1, midpoint_le_midpoint (ihn _).2 ?_⟩ <;>
      /-
        case succ.refine_1
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        x : X
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Membership.mem (Set.Icc (Urysohns.CU.approx n c.r …
        c : Urysohns.CU P
        ⊢ LE.le (Urysohns.CU.approx n c.right.left x) (Urysohns.CU.approx n c.left x)
      -/
      apply approx_le_approx_of_U_sub_C
    /-
      case succ.refine_1.h
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      x : X
      n : Nat
      ihn : ∀ (c : Urysohns.CU P), Membership.mem (Set.Icc (Urysohns.CU.approx n c.r …
      c : Urysohns.CU P
      ⊢ HasSubset.Subset c.left.U c.right.left.C
    -/
    exacts [subset_closure, subset_closure]
    /-
      🎉 no goals
    -/


theorem approx_le_succ (c : CU P) (n : ℕ) (x : X) : c.approx n x ≤ c.approx (n + 1) x := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    n : Nat
    x : X
    ⊢ LE.le (Urysohns.CU.approx n c x) (Urysohns.CU.approx (HAdd.hAdd n 1) c x)
  -/
  induction' n with n ihn generalizing c
    /-
      case zero
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      x : X
      c : Urysohns.CU P
      ⊢ LE.le (Urysohns.CU.approx 0 c x) (Urysohns.CU.approx (HAdd.hAdd 0 1) c x)
    -/
  · simp only [approx, right_U, right_le_midpoint]
    /-
      case zero
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      x : X
      c : Urysohns.CU P
      ⊢ LE.le ((HasCompl.compl c.U).indicator 1 x) ((HasCompl.compl c.left.U).indica …
    -/
    exact (approx_mem_Icc_right_left c 0 x).2
    /-
      🎉 no goals
    -/
    /-
      case succ
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      x : X
      n : Nat
      ihn : ∀ (c : Urysohns.CU P), LE.le (Urysohns.CU.approx n c x) (Urysohns.CU.app …
      c : Urysohns.CU P
      ⊢ LE.le (Urysohns.CU.approx (HAdd.hAdd n 1) c x) (Urysohns.CU.approx (HAdd.hAd …
    -/
  · rw [approx, approx]
    /-
      case succ
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      x : X
      n : Nat
      ihn : ∀ (c : Urysohns.CU P), LE.le (Urysohns.CU.approx n c x) (Urysohns.CU.app …
      c : Urysohns.CU P
      ⊢ LE.le (midpoint Real (Urysohns.CU.approx n c.left x) (Urysohns.CU.approx n c …
    -/
    exact midpoint_le_midpoint (ihn _) (ihn _)
    /-
      🎉 no goals
    -/


theorem approx_mono (c : CU P) (x : X) : Monotone fun n => c.approx n x :=
  monotone_nat_of_le_succ fun n => c.approx_le_succ n x


/-- A continuous function `f : X → ℝ` such that

* `0 ≤ f x ≤ 1` for all `x`;
* `f` equals zero on `c.C` and equals one outside of `c.U`;
-/
protected noncomputable def lim (c : CU P) (x : X) : ℝ :=
  ⨆ n, c.approx n x


theorem tendsto_approx_atTop (c : CU P) (x : X) :
    Tendsto (fun n => c.approx n x) atTop (𝓝 <| c.lim x) :=
  tendsto_atTop_ciSup (c.approx_mono x) ⟨1, fun _ ⟨_, hn⟩ => hn ▸ c.approx_le_one _ _⟩


theorem lim_of_mem_C (c : CU P) (x : X) (h : x ∈ c.C) : c.lim x = 0 := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    x : X
    h : Membership.mem c.C x
    ⊢ Eq (c.lim x) 0
  -/
  simp only [CU.lim, approx_of_mem_C, h, ciSup_const]
  /-
    🎉 no goals
  -/


theorem disjoint_C_support_lim (c : CU P) : Disjoint c.C (Function.support c.lim) :=
  Function.disjoint_support_iff.mpr (fun x hx => lim_of_mem_C c x hx)


theorem lim_of_nmem_U (c : CU P) (x : X) (h : x ∉ c.U) : c.lim x = 1 := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    x : X
    h : Not (Membership.mem c.U x)
    ⊢ Eq (c.lim x) 1
  -/
  simp only [CU.lim, approx_of_nmem_U c _ h, ciSup_const]
  /-
    🎉 no goals
  -/


theorem lim_eq_midpoint (c : CU P) (x : X) :
    c.lim x = midpoint ℝ (c.left.lim x) (c.right.lim x) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    x : X
    ⊢ Eq (c.lim x) (midpoint Real (c.left.lim x) (c.right.lim x))
  -/
  refine tendsto_nhds_unique (c.tendsto_approx_atTop x) ((tendsto_add_atTop_iff_nat 1).1 ?_)
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    x : X
    ⊢ Filter.Tendsto (fun n => Urysohns.CU.approx (HAdd.hAdd n 1) c x) Filter.atTo …
  -/
  simp only [approx]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    x : X
    ⊢ Filter.Tendsto (fun n => midpoint Real (Urysohns.CU.approx n c.left x) (Urys …
  -/
  exact (c.left.tendsto_approx_atTop x).midpoint (c.right.tendsto_approx_atTop x)
  /-
    🎉 no goals
  -/


theorem approx_le_lim (c : CU P) (x : X) (n : ℕ) : c.approx n x ≤ c.lim x :=
  le_ciSup (c.bddAbove_range_approx x) _


theorem lim_nonneg (c : CU P) (x : X) : 0 ≤ c.lim x :=
  (c.approx_nonneg 0 x).trans (c.approx_le_lim x 0)


theorem lim_le_one (c : CU P) (x : X) : c.lim x ≤ 1 :=
  ciSup_le fun _ => c.approx_le_one _ _


theorem lim_mem_Icc (c : CU P) (x : X) : c.lim x ∈ Icc (0 : ℝ) 1 :=
  ⟨c.lim_nonneg x, c.lim_le_one x⟩


/-- Continuity of `Urysohns.CU.lim`. See module docstring for a sketch of the proofs. -/
theorem continuous_lim (c : CU P) : Continuous c.lim := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    ⊢ Continuous c.lim
  -/
  obtain ⟨h0, h1234, h1⟩ : 0 < (2⁻¹ : ℝ) ∧ (2⁻¹ : ℝ) < 3 / 4 ∧ (3 / 4 : ℝ) < 1 := by norm_num
  refine
    continuous_iff_continuousAt.2 fun x =>
      (Metric.nhds_basis_closedBall_pow (h0.trans h1234) h1).tendsto_right_iff.2 fun n _ => ?_
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    h0 : LT.lt 0 (Inv.inv 2)
    h1234 : LT.lt (Inv.inv 2) (3 / 4)
    h1 : LT.lt (3 / 4) 1
    x : X
    n : Nat
    x✝ : True
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Metric.closedBall (c.lim x) (H …
  -/
  simp only [Metric.mem_closedBall]
  /-
    case intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    P : Set X → Prop
    c : Urysohns.CU P
    h0 : LT.lt 0 (Inv.inv 2)
    h1234 : LT.lt (Inv.inv 2) (3 / 4)
    h1 : LT.lt (3 / 4) 1
    x : X
    n : Nat
    x✝ : True
    ⊢ Filter.Eventually (fun x_1 => LE.le (Dist.dist (c.lim x_1) (c.lim x)) (HPow. …
  -/
  induction' n with n ihn generalizing c
    /-
      case intro.intro.zero
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      h0 : LT.lt 0 (Inv.inv 2)
      h1234 : LT.lt (Inv.inv 2) (3 / 4)
      h1 : LT.lt (3 / 4) 1
      x : X
      x✝ : True
      c : Urysohns.CU P
      ⊢ Filter.Eventually (fun x_1 => LE.le (Dist.dist (c.lim x_1) (c.lim x)) (HPow. …
    -/
  · filter_upwards with y
    /-
      case intro.intro.zero.h
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      h0 : LT.lt 0 (Inv.inv 2)
      h1234 : LT.lt (Inv.inv 2) (3 / 4)
      h1 : LT.lt (3 / 4) 1
      x : X
      x✝ : True
      c : Urysohns.CU P
      y : X
      ⊢ LE.le (Dist.dist (c.lim y) (c.lim x)) (HPow.hPow (3 / 4) 0)
    -/
    rw [pow_zero]
    /-
      case intro.intro.zero.h
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      h0 : LT.lt 0 (Inv.inv 2)
      h1234 : LT.lt (Inv.inv 2) (3 / 4)
      h1 : LT.lt (3 / 4) 1
      x : X
      x✝ : True
      c : Urysohns.CU P
      y : X
      ⊢ LE.le (Dist.dist (c.lim y) (c.lim x)) 1
    -/
    exact Real.dist_le_of_mem_Icc_01 (c.lim_mem_Icc _) (c.lim_mem_Icc _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.succ
      X : Type u_1
      inst✝ : TopologicalSpace X
      P : Set X → Prop
      h0 : LT.lt 0 (Inv.inv 2)
      h1234 : LT.lt (Inv.inv 2) (3 / 4)
      h1 : LT.lt (3 / 4) 1
      x : X
      x✝ : True
      n : Nat
      ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
      c : Urysohns.CU P
      ⊢ Filter.Eventually (fun x_1 => LE.le (Dist.dist (c.lim x_1) (c.lim x)) (HPow. …
    -/
  · by_cases hxl : x ∈ c.left.U
      /-
        case pos
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        h0 : LT.lt 0 (Inv.inv 2)
        h1234 : LT.lt (Inv.inv 2) (3 / 4)
        h1 : LT.lt (3 / 4) 1
        x : X
        x✝ : True
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
        c : Urysohns.CU P
        hxl : Membership.mem c.left.U x
        ⊢ Filter.Eventually (fun x_1 => LE.le (Dist.dist (c.lim x_1) (c.lim x)) (HPow. …
      -/
    · filter_upwards [IsOpen.mem_nhds c.left.open_U hxl, ihn c.left] with _ hyl hyd
      rw [pow_succ', c.lim_eq_midpoint, c.lim_eq_midpoint,
        c.right.lim_of_mem_C _ (c.left_U_subset_right_C hyl),
        c.right.lim_of_mem_C _ (c.left_U_subset_right_C hxl)]
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        h0 : LT.lt 0 (Inv.inv 2)
        h1234 : LT.lt (Inv.inv 2) (3 / 4)
        h1 : LT.lt (3 / 4) 1
        x : X
        x✝ : True
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
        c : Urysohns.CU P
        hxl : Membership.mem c.left.U x
        a✝ : X
        hyl : Membership.mem c.left.U a✝
        hyd : LE.le (Dist.dist (c.left.lim a✝) (c.left.lim x)) (HPow.hPow (3 / 4) n)
        ⊢ LE.le (Dist.dist (midpoint Real (c.left.lim a✝) 0) (midpoint Real (c.left.li …
      -/
      refine (dist_midpoint_midpoint_le _ _ _ _).trans ?_
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        h0 : LT.lt 0 (Inv.inv 2)
        h1234 : LT.lt (Inv.inv 2) (3 / 4)
        h1 : LT.lt (3 / 4) 1
        x : X
        x✝ : True
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
        c : Urysohns.CU P
        hxl : Membership.mem c.left.U x
        a✝ : X
        hyl : Membership.mem c.left.U a✝
        hyd : LE.le (Dist.dist (c.left.lim a✝) (c.left.lim x)) (HPow.hPow (3 / 4) n)
        ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (Dist.dist (c.left.lim a✝) (c.left.lim x)) (Dist …
      -/
      rw [dist_self, add_zero, div_eq_inv_mul]
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        h0 : LT.lt 0 (Inv.inv 2)
        h1234 : LT.lt (Inv.inv 2) (3 / 4)
        h1 : LT.lt (3 / 4) 1
        x : X
        x✝ : True
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
        c : Urysohns.CU P
        hxl : Membership.mem c.left.U x
        a✝ : X
        hyl : Membership.mem c.left.U a✝
        hyd : LE.le (Dist.dist (c.left.lim a✝) (c.left.lim x)) (HPow.hPow (3 / 4) n)
        ⊢ LE.le (HMul.hMul (Inv.inv 2) (Dist.dist (c.left.lim a✝) (c.left.lim x))) (HM …
      -/
      gcongr
      /-
        🎉 no goals
      -/
    · replace hxl : x ∈ c.left.right.Cᶜ :=
        compl_subset_compl.2 c.left.right.subset hxl
      filter_upwards [IsOpen.mem_nhds (isOpen_compl_iff.2 c.left.right.closed_C) hxl,
        ihn c.left.right, ihn c.right] with y hyl hydl hydr
      replace hxl : x ∉ c.left.left.U :=
        compl_subset_compl.2 c.left.left_U_subset_right_C hxl
      replace hyl : y ∉ c.left.left.U :=
        compl_subset_compl.2 c.left.left_U_subset_right_C hyl
      simp only [pow_succ, c.lim_eq_midpoint, c.left.lim_eq_midpoint,
        c.left.left.lim_of_nmem_U _ hxl, c.left.left.lim_of_nmem_U _ hyl]
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        h0 : LT.lt 0 (Inv.inv 2)
        h1234 : LT.lt (Inv.inv 2) (3 / 4)
        h1 : LT.lt (3 / 4) 1
        x : X
        x✝ : True
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
        c : Urysohns.CU P
        y : X
        hydl : LE.le (Dist.dist (c.left.right.lim y) (c.left.right.lim x)) (HPow.hPow  …
        hydr : LE.le (Dist.dist (c.right.lim y) (c.right.lim x)) (HPow.hPow (3 / 4) n)
        hxl : Not (Membership.mem c.left.left.U x)
        hyl : Not (Membership.mem c.left.left.U y)
        ⊢ LE.le (Dist.dist (midpoint Real (midpoint Real 1 (c.left.right.lim y)) (c.ri …
      -/
      refine (dist_midpoint_midpoint_le _ _ _ _).trans ?_
      refine (div_le_div_of_nonneg_right (add_le_add_right (dist_midpoint_midpoint_le _ _ _ _) _)
        zero_le_two).trans ?_
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        h0 : LT.lt 0 (Inv.inv 2)
        h1234 : LT.lt (Inv.inv 2) (3 / 4)
        h1 : LT.lt (3 / 4) 1
        x : X
        x✝ : True
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
        c : Urysohns.CU P
        y : X
        hydl : LE.le (Dist.dist (c.left.right.lim y) (c.left.right.lim x)) (HPow.hPow  …
        hydr : LE.le (Dist.dist (c.right.lim y) (c.right.lim x)) (HPow.hPow (3 / 4) n)
        hxl : Not (Membership.mem c.left.left.U x)
        hyl : Not (Membership.mem c.left.left.U y)
        ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (HAdd.hAdd (Dist.dist 1 1) (Dist.dist …
      -/
      rw [dist_self, zero_add]
      /-
        case h
        X : Type u_1
        inst✝ : TopologicalSpace X
        P : Set X → Prop
        h0 : LT.lt 0 (Inv.inv 2)
        h1234 : LT.lt (Inv.inv 2) (3 / 4)
        h1 : LT.lt (3 / 4) 1
        x : X
        x✝ : True
        n : Nat
        ihn : ∀ (c : Urysohns.CU P), Filter.Eventually (fun x_1 => LE.le (Dist.dist (c …
        c : Urysohns.CU P
        y : X
        hydl : LE.le (Dist.dist (c.left.right.lim y) (c.left.right.lim x)) (HPow.hPow  …
        hydr : LE.le (Dist.dist (c.right.lim y) (c.right.lim x)) (HPow.hPow (3 / 4) n)
        hxl : Not (Membership.mem c.left.left.U x)
        hyl : Not (Membership.mem c.left.left.U y)
        ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HDiv.hDiv (Dist.dist (c.left.right.lim y) (c.le …
      -/
      set r := (3 / 4 : ℝ) ^ n
      calc _ ≤ (r / 2 + r) / 2 := by gcongr
        _ = _ := by field_simp; ring


/-- Urysohn's lemma: if `s` and `t` are two disjoint closed sets in a normal topological space `X`,
then there exists a continuous function `f : X → ℝ` such that

* `f` equals zero on `s`;
* `f` equals one on `t`;
* `0 ≤ f x ≤ 1` for all `x`.
-/
theorem exists_continuous_zero_one_of_isClosed [NormalSpace X]
    {s t : Set X} (hs : IsClosed s) (ht : IsClosed t)
    (hd : Disjoint s t) : ∃ f : C(X, ℝ), EqOn f 0 s ∧ EqOn f 1 t ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 := by
  -- The actual proof is in the code above. Here we just repack it into the expected format.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : NormalSpace X
    s t : Set X
    hs : IsClosed s
    ht : IsClosed t
    hd : Disjoint s t
    ⊢ Exists fun f => And (Set.EqOn (⇑f) 0 s) (And (Set.EqOn (⇑f) 1 t) (∀ (x : X), …
  -/
  let P : Set X → Prop := fun _ ↦ True
  set c : Urysohns.CU P :=
  { C := s
    U := tᶜ
    P_C := trivial
    closed_C := hs
    open_U := ht.isOpen_compl
    subset := disjoint_left.1 hd
    hP := by
      rintro c u c_closed - u_open cu
      rcases normal_exists_closure_subset c_closed u_open cu with ⟨v, v_open, cv, hv⟩
      exact ⟨v, v_open, cv, hv, trivial⟩ }
  exact ⟨⟨c.lim, c.continuous_lim⟩, c.lim_of_mem_C, fun x hx => c.lim_of_nmem_U _ fun h => h hx,
    c.lim_mem_Icc⟩


/-- Urysohn's lemma: if `s` and `t` are two disjoint sets in a regular locally compact topological
space `X`, with `s` compact and `t` closed, then there exists a continuous
function `f : X → ℝ` such that

* `f` equals zero on `s`;
* `f` equals one on `t`;
* `0 ≤ f x ≤ 1` for all `x`.
-/
theorem exists_continuous_zero_one_of_isCompact [RegularSpace X] [LocallyCompactSpace X]
    {s t : Set X} (hs : IsCompact s) (ht : IsClosed t) (hd : Disjoint s t) :
    ∃ f : C(X, ℝ), EqOn f 0 s ∧ EqOn f 1 t ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 := by
  obtain ⟨k, k_comp, k_closed, sk, kt⟩ : ∃ k, IsCompact k ∧ IsClosed k ∧ s ⊆ interior k ∧ k ⊆ tᶜ :=
    exists_compact_closed_between hs ht.isOpen_compl hd.symm.subset_compl_left
  /-
    case intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    ht : IsClosed t
    hd : Disjoint s t
    k : Set X
    k_comp : IsCompact k
    k_closed : IsClosed k
    sk : HasSubset.Subset s (interior k)
    kt : HasSubset.Subset k (HasCompl.compl t)
    ⊢ Exists fun f => And (Set.EqOn (⇑f) 0 s) (And (Set.EqOn (⇑f) 1 t) (∀ (x : X), …
  -/
  let P : Set X → Prop := IsCompact
  set c : Urysohns.CU P :=
  { C := k
    U := tᶜ
    P_C := k_comp
    closed_C := k_closed
    open_U := ht.isOpen_compl
    subset := kt
    hP := by
      rintro c u - c_comp u_open cu
      rcases exists_compact_closed_between c_comp u_open cu with ⟨k, k_comp, k_closed, ck, ku⟩
      have A : closure (interior k) ⊆ k :=
        (IsClosed.closure_subset_iff k_closed).2 interior_subset
      refine ⟨interior k, isOpen_interior, ck, A.trans ku,
        k_comp.of_isClosed_subset isClosed_closure A⟩ }
  exact ⟨⟨c.lim, c.continuous_lim⟩, fun x hx ↦ c.lim_of_mem_C _ (sk.trans interior_subset hx),
    fun x hx => c.lim_of_nmem_U _ fun h => h hx, c.lim_mem_Icc⟩


/-- Urysohn's lemma: if `s` and `t` are two disjoint sets in a regular locally compact topological
space `X`, with `s` compact and `t` closed, then there exists a continuous
function `f : X → ℝ` such that

* `f` equals zero on `t`;
* `f` equals one on `s`;
* `0 ≤ f x ≤ 1` for all `x`.
-/
theorem exists_continuous_zero_one_of_isCompact' [RegularSpace X] [LocallyCompactSpace X]
    {s t : Set X} (hs : IsCompact s) (ht : IsClosed t) (hd : Disjoint s t) :
    ∃ f : C(X, ℝ), EqOn f 0 t ∧ EqOn f 1 s ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 := by
  obtain ⟨g, hgs, hgt, (hicc : ∀ x, 0 ≤ g x ∧ g x ≤ 1)⟩ := exists_continuous_zero_one_of_isCompact
    hs ht hd
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    ht : IsClosed t
    hd : Disjoint s t
    g : ContinuousMap X Real
    hgs : Set.EqOn (⇑g) 0 s
    hgt : Set.EqOn (⇑g) 1 t
    hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
    ⊢ Exists fun f => And (Set.EqOn (⇑f) 0 t) (And (Set.EqOn (⇑f) 1 s) (∀ (x : X), …
  -/
  use 1 - g
  /-
    case h
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    ht : IsClosed t
    hd : Disjoint s t
    g : ContinuousMap X Real
    hgs : Set.EqOn (⇑g) 0 s
    hgt : Set.EqOn (⇑g) 1 t
    hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
    ⊢ And (Set.EqOn (⇑(HSub.hSub 1 g)) 0 t) (And (Set.EqOn (⇑(HSub.hSub 1 g)) 1 s) …
  -/
  refine ⟨?_, ?_, ?_⟩
    /-
      case h.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      ⊢ Set.EqOn (⇑(HSub.hSub 1 g)) 0 t
    -/
  · intro x hx
    /-
      case h.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      x : X
      hx : Membership.mem t x
      ⊢ Eq ((HSub.hSub 1 g) x) (0 x)
    -/
    simp only [ContinuousMap.sub_apply, ContinuousMap.one_apply, Pi.zero_apply]
    /-
      case h.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      x : X
      hx : Membership.mem t x
      ⊢ Eq (HSub.hSub 1 (g x)) 0
    -/
    exact sub_eq_zero_of_eq (id (EqOn.symm hgt) hx)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      ⊢ Set.EqOn (⇑(HSub.hSub 1 g)) 1 s
    -/
  · intro x hx
    /-
      case h.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      x : X
      hx : Membership.mem s x
      ⊢ Eq ((HSub.hSub 1 g) x) (1 x)
    -/
    simp only [ContinuousMap.sub_apply, ContinuousMap.one_apply, Pi.one_apply, sub_eq_self]
    /-
      case h.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      x : X
      hx : Membership.mem s x
      ⊢ Eq (g x) 0
    -/
    exact hgs hx
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      ⊢ ∀ (x : X), Membership.mem (Set.Icc 0 1) ((HSub.hSub 1 g) x)
    -/
  · intro x
    /-
      case h.refine_3
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      g : ContinuousMap X Real
      hgs : Set.EqOn (⇑g) 0 s
      hgt : Set.EqOn (⇑g) 1 t
      hicc : ∀ (x : X), And (LE.le 0 (g x)) (LE.le (g x) 1)
      x : X
      ⊢ Membership.mem (Set.Icc 0 1) ((HSub.hSub 1 g) x)
    -/
    simpa [and_comm] using hicc x
    /-
      🎉 no goals
    -/


/-- Urysohn's lemma: if `s` and `t` are two disjoint sets in a regular locally compact topological
space `X`, with `s` compact and `t` closed, then there exists a continuous compactly supported
function `f : X → ℝ` such that

* `f` equals one on `s`;
* `f` equals zero on `t`;
* `0 ≤ f x ≤ 1` for all `x`.
-/
theorem exists_continuous_one_zero_of_isCompact [RegularSpace X] [LocallyCompactSpace X]
    {s t : Set X} (hs : IsCompact s) (ht : IsClosed t) (hd : Disjoint s t) :
    ∃ f : C(X, ℝ), EqOn f 1 s ∧ EqOn f 0 t ∧ HasCompactSupport f ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 := by
  obtain ⟨k, k_comp, k_closed, sk, kt⟩ : ∃ k, IsCompact k ∧ IsClosed k ∧ s ⊆ interior k ∧ k ⊆ tᶜ :=
    exists_compact_closed_between hs ht.isOpen_compl hd.symm.subset_compl_left
  rcases exists_continuous_zero_one_of_isCompact hs isOpen_interior.isClosed_compl
    (disjoint_compl_right_iff_subset.mpr sk) with ⟨⟨f, hf⟩, hfs, hft, h'f⟩
  /-
    case intro.intro.intro.intro.intro.mk.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    ht : IsClosed t
    hd : Disjoint s t
    k : Set X
    k_comp : IsCompact k
    k_closed : IsClosed k
    sk : HasSubset.Subset s (interior k)
    kt : HasSubset.Subset k (HasCompl.compl t)
    f : X → Real
    hf : Continuous f
    hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
    hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
    h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
    ⊢ Exists fun f => And (Set.EqOn (⇑f) 1 s) (And (Set.EqOn (⇑f) 0 t) (And (HasCo …
  -/
  have A : t ⊆ (interior k)ᶜ := subset_compl_comm.mpr (interior_subset.trans kt)
  refine ⟨⟨fun x ↦ 1 - f x, continuous_const.sub hf⟩, fun x hx ↦ by simpa using hfs hx,
    fun x hx ↦ by simpa [sub_eq_zero] using (hft (A hx)).symm, ?_, fun x ↦ ?_⟩
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      ⊢ HasCompactSupport ⇑{ toFun := fun x => HSub.hSub 1 (f x), continuous_toFun : …
    -/
  · apply HasCompactSupport.intro' k_comp k_closed (fun x hx ↦ ?_)
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      x : X
      hx : Not (Membership.mem k x)
      ⊢ Eq ({ toFun := fun x => HSub.hSub 1 (f x), continuous_toFun := ⋯ } x) 0
    -/
    simp only [ContinuousMap.coe_mk, sub_eq_zero]
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      x : X
      hx : Not (Membership.mem k x)
      ⊢ Eq 1 (f x)
    -/
    apply (hft _).symm
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      x : X
      hx : Not (Membership.mem k x)
      ⊢ Membership.mem (HasCompl.compl (interior k)) x
    -/
    contrapose! hx
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      x : X
      hx : Not (Membership.mem (HasCompl.compl (interior k)) x)
      ⊢ Membership.mem k x
    -/
    simp only [mem_compl_iff, not_not] at hx
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      x : X
      hx : Membership.mem (interior k) x
      ⊢ Membership.mem k x
    -/
    exact interior_subset hx
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      x : X
      ⊢ Membership.mem (Set.Icc 0 1) ({ toFun := fun x => HSub.hSub 1 (f x), continu …
    -/
  · have : 0 ≤ f x ∧ f x ≤ 1 := by simpa using h'f x
    /-
      case intro.intro.intro.intro.intro.mk.intro.intro.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      ht : IsClosed t
      hd : Disjoint s t
      k : Set X
      k_comp : IsCompact k
      k_closed : IsClosed k
      sk : HasSubset.Subset s (interior k)
      kt : HasSubset.Subset k (HasCompl.compl t)
      f : X → Real
      hf : Continuous f
      hfs : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 0 s
      hft : Set.EqOn (⇑{ toFun := f, continuous_toFun := hf }) 1 (HasCompl.compl (in …
      h'f : ∀ (x : X), Membership.mem (Set.Icc 0 1) ({ toFun := f, continuous_toFun  …
      A : HasSubset.Subset t (HasCompl.compl (interior k))
      x : X
      this : And (LE.le 0 (f x)) (LE.le (f x) 1)
      ⊢ Membership.mem (Set.Icc 0 1) ({ toFun := fun x => HSub.hSub 1 (f x), continu …
    -/
    simp [this]
    /-
      🎉 no goals
    -/


/-- Urysohn's lemma: if `s` and `t` are two disjoint sets in a regular locally compact topological
space `X`, with `s` compact and `t` closed, then there exists a continuous compactly supported
function `f : X → ℝ` such that

* `f` equals one on `s`;
* `f` equals zero on `t`;
* `0 ≤ f x ≤ 1` for all `x`.

Moreover, if `s` is Gδ, one can ensure that `f ⁻¹ {1}` is exactly `s`.
-/
theorem exists_continuous_one_zero_of_isCompact_of_isGδ [RegularSpace X] [LocallyCompactSpace X]
    {s t : Set X} (hs : IsCompact s) (h's : IsGδ s) (ht : IsClosed t) (hd : Disjoint s t) :
    ∃ f : C(X, ℝ), s = f ⁻¹' {1} ∧ EqOn f 0 t ∧ HasCompactSupport f
      ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    h's : IsGδ s
    ht : IsClosed t
    hd : Disjoint s t
    ⊢ Exists fun f => And (Eq s (Set.preimage (⇑f) (Singleton.singleton 1))) (And  …
  -/
  rcases h's.eq_iInter_nat with ⟨U, U_open, hU⟩
  obtain ⟨m, m_comp, -, sm, mt⟩ : ∃ m, IsCompact m ∧ IsClosed m ∧ s ⊆ interior m ∧ m ⊆ tᶜ :=
    exists_compact_closed_between hs ht.isOpen_compl hd.symm.subset_compl_left
  have A n : ∃ f : C(X, ℝ), EqOn f 1 s ∧ EqOn f 0 (U n ∩ interior m)ᶜ ∧ HasCompactSupport f
      ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 := by
    apply exists_continuous_one_zero_of_isCompact hs
      ((U_open n).inter isOpen_interior).isClosed_compl
    rw [disjoint_compl_right_iff_subset]
    exact subset_inter ((hU.subset.trans (iInter_subset U n))) sm
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    h's : IsGδ s
    ht : IsClosed t
    hd : Disjoint s t
    U : Nat → Set X
    U_open : ∀ (n : Nat), IsOpen (U n)
    hU : Eq s (Set.iInter fun n => U n)
    m : Set X
    m_comp : IsCompact m
    sm : HasSubset.Subset s (interior m)
    mt : HasSubset.Subset m (HasCompl.compl t)
    A : ∀ (n : Nat), Exists fun f => And (Set.EqOn (⇑f) 1 s) (And (Set.EqOn (⇑f) 0 …
    ⊢ Exists fun f => And (Eq s (Set.preimage (⇑f) (Singleton.singleton 1))) (And  …
  -/
  choose f fs fm _hf f_range using A
  obtain ⟨u, u_pos, u_sum, hu⟩ : ∃ (u : ℕ → ℝ), (∀ i, 0 < u i) ∧ Summable u ∧ ∑' i, u i = 1 :=
    ⟨fun n ↦ 1/2/2^n, fun n ↦ by positivity, summable_geometric_two' 1, tsum_geometric_two' 1⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    h's : IsGδ s
    ht : IsClosed t
    hd : Disjoint s t
    U : Nat → Set X
    U_open : ∀ (n : Nat), IsOpen (U n)
    hU : Eq s (Set.iInter fun n => U n)
    m : Set X
    m_comp : IsCompact m
    sm : HasSubset.Subset s (interior m)
    mt : HasSubset.Subset m (HasCompl.compl t)
    f : Nat → ContinuousMap X Real
    fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
    fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
    _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
    f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
    u : Nat → Real
    u_pos : ∀ (i : Nat), LT.lt 0 (u i)
    u_sum : Summable u
    hu : Eq (tsum fun i => u i) 1
    ⊢ Exists fun f => And (Eq s (Set.preimage (⇑f) (Singleton.singleton 1))) (And  …
  -/
  let g : X → ℝ := fun x ↦ ∑' n, u n * f n x
  have hgmc : EqOn g 0 mᶜ := by
    intro x hx
    have B n : f n x = 0 := by
      have : mᶜ ⊆ (U n ∩ interior m)ᶜ := by
        simpa using inter_subset_right.trans interior_subset
      exact fm n (this hx)
    simp [g, B]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    h's : IsGδ s
    ht : IsClosed t
    hd : Disjoint s t
    U : Nat → Set X
    U_open : ∀ (n : Nat), IsOpen (U n)
    hU : Eq s (Set.iInter fun n => U n)
    m : Set X
    m_comp : IsCompact m
    sm : HasSubset.Subset s (interior m)
    mt : HasSubset.Subset m (HasCompl.compl t)
    f : Nat → ContinuousMap X Real
    fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
    fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
    _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
    f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
    u : Nat → Real
    u_pos : ∀ (i : Nat), LT.lt 0 (u i)
    u_sum : Summable u
    hu : Eq (tsum fun i => u i) 1
    g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
    hgmc : Set.EqOn g 0 (HasCompl.compl m)
    ⊢ Exists fun f => And (Eq s (Set.preimage (⇑f) (Singleton.singleton 1))) (And  …
  -/
  have I n x : u n * f n x ≤ u n := mul_le_of_le_one_right (u_pos n).le (f_range n x).2
  have S x : Summable (fun n ↦ u n * f n x) := Summable.of_nonneg_of_le
      (fun n ↦ mul_nonneg (u_pos n).le (f_range n x).1) (fun n ↦ I n x) u_sum
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    s t : Set X
    hs : IsCompact s
    h's : IsGδ s
    ht : IsClosed t
    hd : Disjoint s t
    U : Nat → Set X
    U_open : ∀ (n : Nat), IsOpen (U n)
    hU : Eq s (Set.iInter fun n => U n)
    m : Set X
    m_comp : IsCompact m
    sm : HasSubset.Subset s (interior m)
    mt : HasSubset.Subset m (HasCompl.compl t)
    f : Nat → ContinuousMap X Real
    fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
    fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
    _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
    f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
    u : Nat → Real
    u_pos : ∀ (i : Nat), LT.lt 0 (u i)
    u_sum : Summable u
    hu : Eq (tsum fun i => u i) 1
    g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
    hgmc : Set.EqOn g 0 (HasCompl.compl m)
    I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
    S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
    ⊢ Exists fun f => And (Eq s (Set.preimage (⇑f) (Singleton.singleton 1))) (And  …
  -/
  refine ⟨⟨g, ?_⟩, ?_, hgmc.mono (subset_compl_comm.mp mt), ?_, fun x ↦ ⟨?_, ?_⟩⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      ⊢ Continuous g
    -/
  · apply continuous_tsum (fun n ↦ continuous_const.mul (f n).continuous) u_sum (fun n x ↦ ?_)
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      n : Nat
      x : X
      ⊢ LE.le (Norm.norm (HMul.hMul (u n) ((f n) x))) (u n)
    -/
    simpa [abs_of_nonneg, (u_pos n).le, (f_range n x).1] using I n x
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      ⊢ Eq s (Set.preimage (⇑{ toFun := g, continuous_toFun := ⋯ }) (Singleton.singl …
    -/
  · apply Subset.antisymm (fun x hx ↦ by simp [g, fs _ hx, hu]) ?_
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      ⊢ HasSubset.Subset (Set.preimage (⇑{ toFun := g, continuous_toFun := ⋯ }) (Sin …
    -/
    apply compl_subset_compl.1
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      ⊢ HasSubset.Subset (HasCompl.compl s) (HasCompl.compl (Set.preimage (⇑{ toFun  …
    -/
    intro x hx
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      x : X
      hx : Membership.mem (HasCompl.compl s) x
      ⊢ Membership.mem (HasCompl.compl (Set.preimage (⇑{ toFun := g, continuous_toFu …
    -/
    obtain ⟨n, hn⟩ : ∃ n, x ∉ U n := by simpa [hU] using hx
    /-
      case intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      x : X
      hx : Membership.mem (HasCompl.compl s) x
      n : Nat
      hn : Not (Membership.mem (U n) x)
      ⊢ Membership.mem (HasCompl.compl (Set.preimage (⇑{ toFun := g, continuous_toFu …
    -/
    have fnx : f n x = 0 := fm _ (by simp [hn])
    have : g x < 1 := by
      apply lt_of_lt_of_le ?_ hu.le
      exact tsum_lt_tsum (i := n) (fun i ↦ I i x) (by simp [fnx, u_pos n]) (S x) u_sum
    /-
      case intro
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      x : X
      hx : Membership.mem (HasCompl.compl s) x
      n : Nat
      hn : Not (Membership.mem (U n) x)
      fnx : Eq ((f n) x) 0
      this : LT.lt (g x) 1
      ⊢ Membership.mem (HasCompl.compl (Set.preimage (⇑{ toFun := g, continuous_toFu …
    -/
    simpa using this.ne
    /-
      🎉 no goals
    -/
  · exact HasCompactSupport.of_support_subset_isCompact m_comp
      (Function.support_subset_iff'.mpr hgmc)
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_4
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      x : X
      ⊢ LE.le 0 ({ toFun := g, continuous_toFun := ⋯ } x)
    -/
  · exact tsum_nonneg (fun n ↦ mul_nonneg (u_pos n).le (f_range n x).1)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_5
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      x : X
      ⊢ LE.le ({ toFun := g, continuous_toFun := ⋯ } x) 1
    -/
  · apply le_trans _ hu.le
    /-
      X : Type u_1
      inst✝² : TopologicalSpace X
      inst✝¹ : RegularSpace X
      inst✝ : LocallyCompactSpace X
      s t : Set X
      hs : IsCompact s
      h's : IsGδ s
      ht : IsClosed t
      hd : Disjoint s t
      U : Nat → Set X
      U_open : ∀ (n : Nat), IsOpen (U n)
      hU : Eq s (Set.iInter fun n => U n)
      m : Set X
      m_comp : IsCompact m
      sm : HasSubset.Subset s (interior m)
      mt : HasSubset.Subset m (HasCompl.compl t)
      f : Nat → ContinuousMap X Real
      fs : ∀ (n : Nat), Set.EqOn (⇑(f n)) 1 s
      fm : ∀ (n : Nat), Set.EqOn (⇑(f n)) 0 (HasCompl.compl (Inter.inter (U n) (inte …
      _hf : ∀ (n : Nat), HasCompactSupport ⇑(f n)
      f_range : ∀ (n : Nat) (x : X), Membership.mem (Set.Icc 0 1) ((f n) x)
      u : Nat → Real
      u_pos : ∀ (i : Nat), LT.lt 0 (u i)
      u_sum : Summable u
      hu : Eq (tsum fun i => u i) 1
      g : X → Real := fun x => tsum fun n => HMul.hMul (u n) ((f n) x)
      hgmc : Set.EqOn g 0 (HasCompl.compl m)
      I : ∀ (n : Nat) (x : X), LE.le (HMul.hMul (u n) ((f n) x)) (u n)
      S : ∀ (x : X), Summable fun n => HMul.hMul (u n) ((f n) x)
      x : X
      ⊢ LE.le ({ toFun := g, continuous_toFun := ⋯ } x) (tsum fun i => u i)
    -/
    exact tsum_le_tsum (fun n ↦ I n x) (S x) u_sum
    /-
      🎉 no goals
    -/


/-- A variation of Urysohn's lemma. In a `T2Space X`, for a closed set `t` and a relatively
compact open set `s` such that `t ⊆ s`, there is a continuous function `f` supported in `s`,
`f x = 1` on `t` and `0 ≤ f x ≤ 1`. -/
lemma exists_tsupport_one_of_isOpen_isClosed [T2Space X] {s t : Set X}
    (hs : IsOpen s) (hscp : IsCompact (closure s)) (ht : IsClosed t) (hst : t ⊆ s) :
    ∃ f : C(X, ℝ), tsupport f ⊆ s ∧ EqOn f 1 t ∧ ∀ x, f x ∈ Icc (0 : ℝ) 1 := by
-- separate `sᶜ` and `t` by `u` and `v`.
  /-
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    s t : Set X
    hs : IsOpen s
    hscp : IsCompact (closure s)
    ht : IsClosed t
    hst : HasSubset.Subset t s
    ⊢ Exists fun f => And (HasSubset.Subset (tsupport ⇑f) s) (And (Set.EqOn (⇑f) 1 …
  -/
  rw [← compl_compl s] at hscp
  obtain ⟨u, v, huIsOpen, hvIsOpen, hscompl_subset_u, ht_subset_v, hDjsjointuv⟩ :=
    SeparatedNhds.of_isClosed_isCompact_closure_compl_isClosed (isClosed_compl_iff.mpr hs)
    hscp ht (HasSubset.Subset.disjoint_compl_left hst)
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    s t : Set X
    hs : IsOpen s
    hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
    ht : IsClosed t
    hst : HasSubset.Subset t s
    u v : Set X
    huIsOpen : IsOpen u
    hvIsOpen : IsOpen v
    hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
    ht_subset_v : HasSubset.Subset t v
    hDjsjointuv : Disjoint u v
    ⊢ Exists fun f => And (HasSubset.Subset (tsupport ⇑f) s) (And (Set.EqOn (⇑f) 1 …
  -/
  rw [← subset_compl_iff_disjoint_right] at hDjsjointuv
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    s t : Set X
    hs : IsOpen s
    hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
    ht : IsClosed t
    hst : HasSubset.Subset t s
    u v : Set X
    huIsOpen : IsOpen u
    hvIsOpen : IsOpen v
    hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
    ht_subset_v : HasSubset.Subset t v
    hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
    ⊢ Exists fun f => And (HasSubset.Subset (tsupport ⇑f) s) (And (Set.EqOn (⇑f) 1 …
  -/
  have huvc : closure u ⊆ vᶜ := closure_minimal hDjsjointuv hvIsOpen.isClosed_compl
-- although `sᶜ` is not compact, `closure s` is compact and we can apply
-- `SeparatedNhds.of_isClosed_isCompact_closure_compl_isClosed`. To apply the condition
-- recursively, we need to make sure that `sᶜ ⊆ C`.
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    s t : Set X
    hs : IsOpen s
    hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
    ht : IsClosed t
    hst : HasSubset.Subset t s
    u v : Set X
    huIsOpen : IsOpen u
    hvIsOpen : IsOpen v
    hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
    ht_subset_v : HasSubset.Subset t v
    hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
    huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
    ⊢ Exists fun f => And (HasSubset.Subset (tsupport ⇑f) s) (And (Set.EqOn (⇑f) 1 …
  -/
  let P : Set X → Prop := fun C => sᶜ ⊆ C
  set c : Urysohns.CU P :=
  { C := closure u
    U := tᶜ
    P_C := hscompl_subset_u.trans subset_closure
    closed_C := isClosed_closure
    open_U := ht.isOpen_compl
    subset := subset_compl_comm.mp
      (Subset.trans ht_subset_v (subset_compl_comm.mp huvc))
    hP := by
      intro c u0 cIsClosed Pc u0IsOpen csubu0
      obtain ⟨u1, hu1⟩ := SeparatedNhds.of_isClosed_isCompact_closure_compl_isClosed cIsClosed
        (IsCompact.of_isClosed_subset hscp isClosed_closure
        (closure_mono (compl_subset_compl.mpr Pc)))
        (isClosed_compl_iff.mpr u0IsOpen) (HasSubset.Subset.disjoint_compl_right csubu0)
      simp_rw [← subset_compl_iff_disjoint_right, compl_subset_comm (s := u0)] at hu1
      obtain ⟨v1, hu1, hv1, hcu1, hv1u, hu1v1⟩ := hu1
      refine ⟨u1, hu1, hcu1, ?_, (Pc.trans hcu1).trans subset_closure⟩
      exact closure_minimal hu1v1 hv1.isClosed_compl |>.trans hv1u }
-- `c.lim = 0` on `closure u` and `c.lim = 1` on `t`, so that `tsupport c.lim ⊆ s`.
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    s t : Set X
    hs : IsOpen s
    hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
    ht : IsClosed t
    hst : HasSubset.Subset t s
    u v : Set X
    huIsOpen : IsOpen u
    hvIsOpen : IsOpen v
    hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
    ht_subset_v : HasSubset.Subset t v
    hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
    huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
    P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
    c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
    ⊢ Exists fun f => And (HasSubset.Subset (tsupport ⇑f) s) (And (Set.EqOn (⇑f) 1 …
  -/
  use ⟨c.lim, c.continuous_lim⟩
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    s t : Set X
    hs : IsOpen s
    hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
    ht : IsClosed t
    hst : HasSubset.Subset t s
    u v : Set X
    huIsOpen : IsOpen u
    hvIsOpen : IsOpen v
    hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
    ht_subset_v : HasSubset.Subset t v
    hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
    huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
    P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
    c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
    ⊢ And (HasSubset.Subset (tsupport ⇑{ toFun := c.lim, continuous_toFun := ⋯ })  …
  -/
  simp only [ContinuousMap.coe_mk]
  /-
    case h
    X : Type u_1
    inst✝¹ : TopologicalSpace X
    inst✝ : T2Space X
    s t : Set X
    hs : IsOpen s
    hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
    ht : IsClosed t
    hst : HasSubset.Subset t s
    u v : Set X
    huIsOpen : IsOpen u
    hvIsOpen : IsOpen v
    hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
    ht_subset_v : HasSubset.Subset t v
    hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
    huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
    P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
    c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
    ⊢ And (HasSubset.Subset (tsupport c.lim) s) (And (Set.EqOn c.lim 1 t) (∀ (x :  …
  -/
  refine ⟨?_, ?_, Urysohns.CU.lim_mem_Icc c⟩
    /-
      case h.refine_1
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T2Space X
      s t : Set X
      hs : IsOpen s
      hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
      ht : IsClosed t
      hst : HasSubset.Subset t s
      u v : Set X
      huIsOpen : IsOpen u
      hvIsOpen : IsOpen v
      hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
      ht_subset_v : HasSubset.Subset t v
      hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
      huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
      P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
      c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
      ⊢ HasSubset.Subset (tsupport c.lim) s
    -/
  · apply Subset.trans _ (compl_subset_comm.mp hscompl_subset_u)
    /-
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T2Space X
      s t : Set X
      hs : IsOpen s
      hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
      ht : IsClosed t
      hst : HasSubset.Subset t s
      u v : Set X
      huIsOpen : IsOpen u
      hvIsOpen : IsOpen v
      hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
      ht_subset_v : HasSubset.Subset t v
      hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
      huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
      P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
      c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
      ⊢ HasSubset.Subset (tsupport c.lim) (HasCompl.compl u)
    -/
    rw [← IsClosed.closure_eq (isClosed_compl_iff.mpr huIsOpen)]
    /-
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T2Space X
      s t : Set X
      hs : IsOpen s
      hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
      ht : IsClosed t
      hst : HasSubset.Subset t s
      u v : Set X
      huIsOpen : IsOpen u
      hvIsOpen : IsOpen v
      hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
      ht_subset_v : HasSubset.Subset t v
      hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
      huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
      P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
      c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
      ⊢ HasSubset.Subset (tsupport c.lim) (closure (HasCompl.compl u))
    -/
    apply closure_mono
    exact Disjoint.subset_compl_right (disjoint_of_subset_right subset_closure
      (Disjoint.symm (Urysohns.CU.disjoint_C_support_lim c)))
    /-
      case h.refine_2
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T2Space X
      s t : Set X
      hs : IsOpen s
      hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
      ht : IsClosed t
      hst : HasSubset.Subset t s
      u v : Set X
      huIsOpen : IsOpen u
      hvIsOpen : IsOpen v
      hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
      ht_subset_v : HasSubset.Subset t v
      hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
      huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
      P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
      c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
      ⊢ Set.EqOn c.lim 1 t
    -/
  · intro x hx
    /-
      case h.refine_2
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T2Space X
      s t : Set X
      hs : IsOpen s
      hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
      ht : IsClosed t
      hst : HasSubset.Subset t s
      u v : Set X
      huIsOpen : IsOpen u
      hvIsOpen : IsOpen v
      hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
      ht_subset_v : HasSubset.Subset t v
      hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
      huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
      P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
      c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
      x : X
      hx : Membership.mem t x
      ⊢ Eq (c.lim x) (1 x)
    -/
    apply Urysohns.CU.lim_of_nmem_U
    /-
      case h.refine_2.h
      X : Type u_1
      inst✝¹ : TopologicalSpace X
      inst✝ : T2Space X
      s t : Set X
      hs : IsOpen s
      hscp : IsCompact (closure (HasCompl.compl (HasCompl.compl s)))
      ht : IsClosed t
      hst : HasSubset.Subset t s
      u v : Set X
      huIsOpen : IsOpen u
      hvIsOpen : IsOpen v
      hscompl_subset_u : HasSubset.Subset (HasCompl.compl s) u
      ht_subset_v : HasSubset.Subset t v
      hDjsjointuv : HasSubset.Subset u (HasCompl.compl v)
      huvc : HasSubset.Subset (closure u) (HasCompl.compl v)
      P : Set X → Prop := fun C => HasSubset.Subset (HasCompl.compl s) C
      c : Urysohns.CU P := { C := closure u, U := HasCompl.compl t, P_C := ⋯, closed …
      x : X
      hx : Membership.mem t x
      ⊢ Not (Membership.mem c.U x)
    -/
    exact not_mem_compl_iff.mpr hx
    /-
      🎉 no goals
    -/


theorem exists_continuous_nonneg_pos [RegularSpace X] [LocallyCompactSpace X] (x : X) :
    ∃ f : C(X, ℝ), HasCompactSupport f ∧ 0 ≤ (f : X → ℝ) ∧ f x ≠ 0 := by
  /-
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    x : X
    ⊢ Exists fun f => And (HasCompactSupport ⇑f) (And (LE.le 0 ⇑f) (Ne (f x) 0))
  -/
  rcases exists_compact_mem_nhds x with ⟨k, hk, k_mem⟩
  rcases exists_continuous_one_zero_of_isCompact hk isClosed_empty (disjoint_empty k)
    with ⟨f, fk, -, f_comp, hf⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    x : X
    k : Set X
    hk : IsCompact k
    k_mem : Membership.mem (nhds x) k
    f : ContinuousMap X Real
    fk : Set.EqOn (⇑f) 1 k
    f_comp : HasCompactSupport ⇑f
    hf : ∀ (x : X), Membership.mem (Set.Icc 0 1) (f x)
    ⊢ Exists fun f => And (HasCompactSupport ⇑f) (And (LE.le 0 ⇑f) (Ne (f x) 0))
  -/
  refine ⟨f, f_comp, fun x ↦ (hf x).1, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    x : X
    k : Set X
    hk : IsCompact k
    k_mem : Membership.mem (nhds x) k
    f : ContinuousMap X Real
    fk : Set.EqOn (⇑f) 1 k
    f_comp : HasCompactSupport ⇑f
    hf : ∀ (x : X), Membership.mem (Set.Icc 0 1) (f x)
    ⊢ Ne (f x) 0
  -/
  have := fk (mem_of_mem_nhds k_mem)
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    x : X
    k : Set X
    hk : IsCompact k
    k_mem : Membership.mem (nhds x) k
    f : ContinuousMap X Real
    fk : Set.EqOn (⇑f) 1 k
    f_comp : HasCompactSupport ⇑f
    hf : ∀ (x : X), Membership.mem (Set.Icc 0 1) (f x)
    this : Eq (f x) (1 x)
    ⊢ Ne (f x) 0
  -/
  simp only [ContinuousMap.coe_mk, Pi.one_apply] at this
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝² : TopologicalSpace X
    inst✝¹ : RegularSpace X
    inst✝ : LocallyCompactSpace X
    x : X
    k : Set X
    hk : IsCompact k
    k_mem : Membership.mem (nhds x) k
    f : ContinuousMap X Real
    fk : Set.EqOn (⇑f) 1 k
    f_comp : HasCompactSupport ⇑f
    hf : ∀ (x : X), Membership.mem (Set.Icc 0 1) (f x)
    this : Eq (f x) 1
    ⊢ Ne (f x) 0
  -/
  simp [this]
  /-
    🎉 no goals
  -/

