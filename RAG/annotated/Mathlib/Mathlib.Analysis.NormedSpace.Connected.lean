/-- In a real vector space of dimension `> 1`, the complement of any countable set is path
connected. -/
theorem Set.Countable.isPathConnected_compl_of_one_lt_rank
    (h : 1 < Module.rank ℝ E) {s : Set E} (hs : s.Countable) :
    IsPathConnected sᶜ := by
  /-
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    ⊢ IsPathConnected (HasCompl.compl s)
  -/
  have : Nontrivial E := (rank_pos_iff_nontrivial (R := ℝ)).1 (zero_lt_one.trans h)
  -- the set `sᶜ` is dense, therefore nonempty. Pick `a ∈ sᶜ`. We have to show that any
  -- `b ∈ sᶜ` can be joined to `a`.
  /-
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    ⊢ IsPathConnected (HasCompl.compl s)
  -/
  obtain ⟨a, ha⟩ : sᶜ.Nonempty := (hs.dense_compl ℝ).nonempty
  /-
    case intro
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    ⊢ IsPathConnected (HasCompl.compl s)
  -/
  refine ⟨a, ha, ?_⟩
  /-
    case intro
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    ⊢ ∀ {y : E}, Membership.mem (HasCompl.compl s) y → JoinedIn (HasCompl.compl s) …
  -/
  intro b hb
  /-
    case intro
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    b : E
    hb : Membership.mem (HasCompl.compl s) b
    ⊢ JoinedIn (HasCompl.compl s) a b
  -/
  rcases eq_or_ne a b with rfl|hab
    /-
      case intro.inl
      E : Type u_1
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      inst✝² : TopologicalSpace E
      inst✝¹ : ContinuousAdd E
      inst✝ : ContinuousSMul Real E
      h : LT.lt 1 (Module.rank Real E)
      s : Set E
      hs : s.Countable
      this : Nontrivial E
      a : E
      ha hb : Membership.mem (HasCompl.compl s) a
      ⊢ JoinedIn (HasCompl.compl s) a a
    -/
  · exact JoinedIn.refl ha
    /-
      🎉 no goals
    -/
  /- Assume `b ≠ a`. Write `a = c - x` and `b = c + x` for some nonzero `x`. Choose `y` which
  is linearly independent from `x`. Then the segments joining `a = c - x` to `c + ty` are pairwise
  disjoint for varying `t` (except for the endpoint `a`) so only countably many of them can
  intersect `s`. In the same way, there are countably many `t`s for which the segment
  from `b = c + x` to `c + ty` intersects `s`. Choosing `t` outside of these countable exceptions,
  one gets a path in the complement of `s` from `a` to `z = c + ty` and then to `b`.
  -/
  /-
    case intro.inr
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    b : E
    hb : Membership.mem (HasCompl.compl s) b
    hab : Ne a b
    ⊢ JoinedIn (HasCompl.compl s) a b
  -/
  let c := (2 : ℝ)⁻¹ • (a + b)
  /-
    case intro.inr
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    b : E
    hb : Membership.mem (HasCompl.compl s) b
    hab : Ne a b
    c : E := HSMul.hSMul (Inv.inv 2) (HAdd.hAdd a b)
    ⊢ JoinedIn (HasCompl.compl s) a b
  -/
  let x := (2 : ℝ)⁻¹ • (b - a)
  have Ia : c - x = a := by
    simp only [c, x]
    module
  have Ib : c + x = b := by
    simp only [c, x]
    module
  /-
    case intro.inr
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    b : E
    hb : Membership.mem (HasCompl.compl s) b
    hab : Ne a b
    c : E := HSMul.hSMul (Inv.inv 2) (HAdd.hAdd a b)
    x : E := HSMul.hSMul (Inv.inv 2) (HSub.hSub b a)
    Ia : Eq (HSub.hSub c x) a
    Ib : Eq (HAdd.hAdd c x) b
    ⊢ JoinedIn (HasCompl.compl s) a b
  -/
  have x_ne_zero : x ≠ 0 := by simpa [x] using sub_ne_zero.2 hab.symm
  obtain ⟨y, hy⟩ : ∃ y, LinearIndependent ℝ ![x, y] :=
    exists_linearIndependent_pair_of_one_lt_rank h x_ne_zero
  have A : Set.Countable {t : ℝ | ([c + x -[ℝ] c + t • y] ∩ s).Nonempty} := by
    apply countable_setOf_nonempty_of_disjoint _ (fun t ↦ inter_subset_right) hs
    intro t t' htt'
    apply disjoint_iff_inter_eq_empty.2
    have N : {c + x} ∩ s = ∅ := by
      simpa only [singleton_inter_eq_empty, mem_compl_iff, Ib] using hb
    rw [inter_assoc, inter_comm s, inter_assoc, inter_self, ← inter_assoc, ← subset_empty_iff, ← N]
    apply inter_subset_inter_left
    apply Eq.subset
    apply segment_inter_eq_endpoint_of_linearIndependent_of_ne hy htt'.symm
  have B : Set.Countable {t : ℝ | ([c - x -[ℝ] c + t • y] ∩ s).Nonempty} := by
    apply countable_setOf_nonempty_of_disjoint _ (fun t ↦ inter_subset_right) hs
    intro t t' htt'
    apply disjoint_iff_inter_eq_empty.2
    have N : {c - x} ∩ s = ∅ := by
      simpa only [singleton_inter_eq_empty, mem_compl_iff, Ia] using ha
    rw [inter_assoc, inter_comm s, inter_assoc, inter_self, ← inter_assoc, ← subset_empty_iff, ← N]
    apply inter_subset_inter_left
    rw [sub_eq_add_neg _ x]
    apply Eq.subset
    apply segment_inter_eq_endpoint_of_linearIndependent_of_ne _ htt'.symm
    convert hy.units_smul ![-1, 1]
    simp [← List.ofFn_inj]
  obtain ⟨t, ht⟩ : Set.Nonempty ({t : ℝ | ([c + x -[ℝ] c + t • y] ∩ s).Nonempty}
      ∪ {t : ℝ | ([c - x -[ℝ] c + t • y] ∩ s).Nonempty})ᶜ := ((A.union B).dense_compl ℝ).nonempty
  /-
    case intro.inr.intro.intro
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    b : E
    hb : Membership.mem (HasCompl.compl s) b
    hab : Ne a b
    c : E := HSMul.hSMul (Inv.inv 2) (HAdd.hAdd a b)
    x : E := HSMul.hSMul (Inv.inv 2) (HSub.hSub b a)
    Ia : Eq (HSub.hSub c x) a
    Ib : Eq (HAdd.hAdd c x) b
    x_ne_zero : Ne x 0
    y : E
    hy : LinearIndependent Real (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpt …
    A : (setOf fun t => (Inter.inter (segment Real (HAdd.hAdd c x) (HAdd.hAdd c (H …
    B : (setOf fun t => (Inter.inter (segment Real (HSub.hSub c x) (HAdd.hAdd c (H …
    t : Real
    ht : Membership.mem (HasCompl.compl (Union.union (setOf fun t => (Inter.inter  …
    ⊢ JoinedIn (HasCompl.compl s) a b
  -/
  let z := c + t • y
  simp only [compl_union, mem_inter_iff, mem_compl_iff, mem_setOf_eq, not_nonempty_iff_eq_empty]
    at ht
  have JA : JoinedIn sᶜ a z := by
    apply JoinedIn.of_segment_subset
    rw [subset_compl_iff_disjoint_right, disjoint_iff_inter_eq_empty]
    convert ht.2
    exact Ia.symm
  have JB : JoinedIn sᶜ b z := by
    apply JoinedIn.of_segment_subset
    rw [subset_compl_iff_disjoint_right, disjoint_iff_inter_eq_empty]
    convert ht.1
    exact Ib.symm
  /-
    case intro.inr.intro.intro
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : ContinuousAdd E
    inst✝ : ContinuousSMul Real E
    h : LT.lt 1 (Module.rank Real E)
    s : Set E
    hs : s.Countable
    this : Nontrivial E
    a : E
    ha : Membership.mem (HasCompl.compl s) a
    b : E
    hb : Membership.mem (HasCompl.compl s) b
    hab : Ne a b
    c : E := HSMul.hSMul (Inv.inv 2) (HAdd.hAdd a b)
    x : E := HSMul.hSMul (Inv.inv 2) (HSub.hSub b a)
    Ia : Eq (HSub.hSub c x) a
    Ib : Eq (HAdd.hAdd c x) b
    x_ne_zero : Ne x 0
    y : E
    hy : LinearIndependent Real (Matrix.vecCons x (Matrix.vecCons y Matrix.vecEmpt …
    A : (setOf fun t => (Inter.inter (segment Real (HAdd.hAdd c x) (HAdd.hAdd c (H …
    B : (setOf fun t => (Inter.inter (segment Real (HSub.hSub c x) (HAdd.hAdd c (H …
    t : Real
    z : E := HAdd.hAdd c (HSMul.hSMul t y)
    ht : And (Eq (Inter.inter (segment Real (HAdd.hAdd c x) (HAdd.hAdd c (HSMul.hS …
    JA : JoinedIn (HasCompl.compl s) a z
    JB : JoinedIn (HasCompl.compl s) b z
    ⊢ JoinedIn (HasCompl.compl s) a b
  -/
  exact JA.trans JB.symm
  /-
    🎉 no goals
  -/


/-- In a real vector space of dimension `> 1`, the complement of any countable set is
connected. -/
theorem Set.Countable.isConnected_compl_of_one_lt_rank (h : 1 < Module.rank ℝ E) {s : Set E}
    (hs : s.Countable) : IsConnected sᶜ :=
  (hs.isPathConnected_compl_of_one_lt_rank h).isConnected


/-- In a real vector space of dimension `> 1`, the complement of any singleton is path-connected. -/
theorem isPathConnected_compl_singleton_of_one_lt_rank (h : 1 < Module.rank ℝ E) (x : E) :
    IsPathConnected {x}ᶜ :=
  Set.Countable.isPathConnected_compl_of_one_lt_rank h (countable_singleton x)


/-- In a real vector space of dimension `> 1`, the complement of a singleton is connected. -/
theorem isConnected_compl_singleton_of_one_lt_rank (h : 1 < Module.rank ℝ E) (x : E) :
    IsConnected {x}ᶜ :=
  (isPathConnected_compl_singleton_of_one_lt_rank h x).isConnected


/-- In a real vector space of dimension `> 1`, any sphere of nonnegative radius is
path connected. -/
theorem isPathConnected_sphere (h : 1 < Module.rank ℝ E) (x : E) {r : ℝ} (hr : 0 ≤ r) :
    IsPathConnected (sphere x r) := by
  /- when `r > 0`, we write the sphere as the image of `{0}ᶜ` under the map
  `y ↦ x + (r * ‖y‖⁻¹) • y`. Since the image under a continuous map of a path connected set
  is path connected, this concludes the proof. -/
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : LT.lt 1 (Module.rank Real E)
    x : E
    r : Real
    hr : LE.le 0 r
    ⊢ IsPathConnected (Metric.sphere x r)
  -/
  rcases hr.eq_or_lt with rfl|rpos
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      h : LT.lt 1 (Module.rank Real E)
      x : E
      hr : LE.le 0 0
      ⊢ IsPathConnected (Metric.sphere x 0)
    -/
  · simpa using isPathConnected_singleton x
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : LT.lt 1 (Module.rank Real E)
    x : E
    r : Real
    hr : LE.le 0 r
    rpos : LT.lt 0 r
    ⊢ IsPathConnected (Metric.sphere x r)
  -/
  let f : E → E := fun y ↦ x + (r * ‖y‖⁻¹) • y
  have A : ContinuousOn f {0}ᶜ := by
    intro y hy
    apply (continuousAt_const.add _).continuousWithinAt
    apply (continuousAt_const.mul (ContinuousAt.inv₀ continuousAt_id.norm ?_)).smul continuousAt_id
    simpa using hy
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : LT.lt 1 (Module.rank Real E)
    x : E
    r : Real
    hr : LE.le 0 r
    rpos : LT.lt 0 r
    f : E → E := fun y => HAdd.hAdd x (HSMul.hSMul (HMul.hMul r (Inv.inv (Norm.nor …
    A : ContinuousOn f (HasCompl.compl (Singleton.singleton 0))
    ⊢ IsPathConnected (Metric.sphere x r)
  -/
  have B : IsPathConnected ({0}ᶜ : Set E) := isPathConnected_compl_singleton_of_one_lt_rank h 0
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : LT.lt 1 (Module.rank Real E)
    x : E
    r : Real
    hr : LE.le 0 r
    rpos : LT.lt 0 r
    f : E → E := fun y => HAdd.hAdd x (HSMul.hSMul (HMul.hMul r (Inv.inv (Norm.nor …
    A : ContinuousOn f (HasCompl.compl (Singleton.singleton 0))
    B : IsPathConnected (HasCompl.compl (Singleton.singleton 0))
    ⊢ IsPathConnected (Metric.sphere x r)
  -/
  have C : IsPathConnected (f '' {0}ᶜ) := B.image' A
  have : f '' {0}ᶜ = sphere x r := by
    apply Subset.antisymm
    · rintro - ⟨y, hy, rfl⟩
      have : ‖y‖ ≠ 0 := by simpa using hy
      simp [f, norm_smul, abs_of_nonneg hr, mul_assoc, inv_mul_cancel₀ this]
    · intro y hy
      refine ⟨y - x, ?_, ?_⟩
      · intro H
        simp only [mem_singleton_iff, sub_eq_zero] at H
        simp only [H, mem_sphere_iff_norm, sub_self, norm_zero] at hy
        exact rpos.ne hy
      · simp [f, mem_sphere_iff_norm.1 hy, mul_inv_cancel₀ rpos.ne']
  /-
    case inr
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : LT.lt 1 (Module.rank Real E)
    x : E
    r : Real
    hr : LE.le 0 r
    rpos : LT.lt 0 r
    f : E → E := fun y => HAdd.hAdd x (HSMul.hSMul (HMul.hMul r (Inv.inv (Norm.nor …
    A : ContinuousOn f (HasCompl.compl (Singleton.singleton 0))
    B : IsPathConnected (HasCompl.compl (Singleton.singleton 0))
    C : IsPathConnected (Set.image f (HasCompl.compl (Singleton.singleton 0)))
    this : Eq (Set.image f (HasCompl.compl (Singleton.singleton 0))) (Metric.spher …
    ⊢ IsPathConnected (Metric.sphere x r)
  -/
  rwa [this] at C
  /-
    🎉 no goals
  -/


/-- In a real vector space of dimension `> 1`, any sphere of nonnegative radius is connected. -/
theorem isConnected_sphere (h : 1 < Module.rank ℝ E) (x : E) {r : ℝ} (hr : 0 ≤ r) :
    IsConnected (sphere x r) :=
  (isPathConnected_sphere h x hr).isConnected


/-- In a real vector space of dimension `> 1`, any sphere is preconnected. -/
theorem isPreconnected_sphere (h : 1 < Module.rank ℝ E) (x : E) (r : ℝ) :
    IsPreconnected (sphere x r) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    h : LT.lt 1 (Module.rank Real E)
    x : E
    r : Real
    ⊢ IsPreconnected (Metric.sphere x r)
  -/
  rcases le_or_lt 0 r with hr|hr
    /-
      case inl
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      h : LT.lt 1 (Module.rank Real E)
      x : E
      r : Real
      hr : LE.le 0 r
      ⊢ IsPreconnected (Metric.sphere x r)
    -/
  · exact (isConnected_sphere h x hr).isPreconnected
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      h : LT.lt 1 (Module.rank Real E)
      x : E
      r : Real
      hr : LT.lt r 0
      ⊢ IsPreconnected (Metric.sphere x r)
    -/
  · simpa [hr] using isPreconnected_empty
    /-
      🎉 no goals
    -/


/-- Let `E` be a linear subspace in a real vector space.
If `E` has codimension at least two, its complement is path-connected. -/
theorem isPathConnected_compl_of_one_lt_codim {E : Submodule ℝ F}
    (hcodim : 1 < Module.rank ℝ (F ⧸ E)) : IsPathConnected (Eᶜ : Set F) := by
  /-
    F : Type u_1
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module Real F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul Real F
    E : Submodule Real F
    hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
    ⊢ IsPathConnected (HasCompl.compl ↑E)
  -/
  rcases E.exists_isCompl with ⟨E', hE'⟩
  refine isPathConnected_compl_of_isPathConnected_compl_zero hE'.symm
    (isPathConnected_compl_singleton_of_one_lt_rank ?_ 0)
  /-
    case intro
    F : Type u_1
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module Real F
    inst✝² : TopologicalSpace F
    inst✝¹ : TopologicalAddGroup F
    inst✝ : ContinuousSMul Real F
    E : Submodule Real F
    hcodim : LT.lt 1 (Module.rank Real (HasQuotient.Quotient F E))
    E' : Submodule Real F
    hE' : IsCompl E E'
    ⊢ LT.lt 1 (Module.rank Real (Subtype fun x => Membership.mem E' x))
  -/
  rwa [← (E.quotientEquivOfIsCompl E' hE').rank_eq]
  /-
    🎉 no goals
  -/


/-- Let `E` be a linear subspace in a real vector space.
If `E` has codimension at least two, its complement is connected. -/
theorem isConnected_compl_of_one_lt_codim {E : Submodule ℝ F} (hcodim : 1 < Module.rank ℝ (F ⧸ E)) :
    IsConnected (Eᶜ : Set F) :=
  (isPathConnected_compl_of_one_lt_codim hcodim).isConnected


theorem Submodule.connectedComponentIn_eq_self_of_one_lt_codim (E : Submodule ℝ F)
    (hcodim : 1 < Module.rank ℝ (F ⧸ E)) {x : F} (hx : x ∉ E) :
    connectedComponentIn ((E : Set F)ᶜ) x = (E : Set F)ᶜ :=
  (isConnected_compl_of_one_lt_codim hcodim).2.connectedComponentIn hx


