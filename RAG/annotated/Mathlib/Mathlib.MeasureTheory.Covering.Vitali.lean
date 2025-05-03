/-- **Vitali covering theorem**: given a set `t` of subsets of a type, one may extract a disjoint
subfamily `u` such that the `τ`-enlargement of this family covers all elements of `t`, where `τ > 1`
is any fixed number.

When `t` is a family of balls, the `τ`-enlargement of `ball x r` is `ball x ((1+2τ) r)`. In general,
it is expressed in terms of a function `δ` (think "radius" or "diameter"), positive and bounded on
all elements of `t`. The condition is that every element `a` of `t` should intersect an
element `b` of `u` of size larger than that of `a` up to `τ`, i.e., `δ b ≥ δ a / τ`.

We state the lemma slightly more generally, with an indexed family of sets `B a` for `a ∈ t`, for
wider applicability.
-/
theorem exists_disjoint_subfamily_covering_enlargement (B : ι → Set α) (t : Set ι) (δ : ι → ℝ)
    (τ : ℝ) (hτ : 1 < τ) (δnonneg : ∀ a ∈ t, 0 ≤ δ a) (R : ℝ) (δle : ∀ a ∈ t, δ a ≤ R)
    (hne : ∀ a ∈ t, (B a).Nonempty) :
    ∃ u ⊆ t,
      u.PairwiseDisjoint B ∧ ∀ a ∈ t, ∃ b ∈ u, (B a ∩ B b).Nonempty ∧ δ a ≤ τ * δ b := by
  /- The proof could be formulated as a transfinite induction. First pick an element of `t` with `δ`
  as large as possible (up to a factor of `τ`). Then among the remaining elements not intersecting
  the already chosen one, pick another element with large `δ`. Go on forever (transfinitely) until
  there is nothing left.

  Instead, we give a direct Zorn-based argument. Consider a maximal family `u` of disjoint sets
  with the following property: if an element `a` of `t` intersects some element `b` of `u`, then it
  intersects some `b' ∈ u` with `δ b' ≥ δ a / τ`. Such a maximal family exists by Zorn. If this
  family did not intersect some element `a ∈ t`, then take an element `a' ∈ t` which does not
  intersect any element of `u`, with `δ a'` almost as large as possible. One checks easily
  that `u ∪ {a'}` still has this property, contradicting the maximality. Therefore, `u`
  intersects all elements of `t`, and by definition it satisfies all the desired properties.
  -/
  let T : Set (Set ι) := { u | u ⊆ t ∧ u.PairwiseDisjoint B ∧
    ∀ a ∈ t, ∀ b ∈ u, (B a ∩ B b).Nonempty → ∃ c ∈ u, (B a ∩ B c).Nonempty ∧ δ a ≤ τ * δ c }
  -- By Zorn, choose a maximal family in the good set `T` of disjoint families.
  obtain ⟨u, hu⟩ : ∃ m, Maximal (fun x ↦ x ∈ T) m := by
    refine zorn_subset _ fun U UT hU => ?_
    refine ⟨⋃₀ U, ?_, fun s hs => subset_sUnion_of_mem hs⟩
    simp only [T, Set.sUnion_subset_iff, and_imp, exists_prop, forall_exists_index, mem_sUnion,
      Set.mem_setOf_eq]
    refine
      ⟨fun u hu => (UT hu).1, (pairwiseDisjoint_sUnion hU.directedOn).2 fun u hu => (UT hu).2.1,
        fun a hat b u uU hbu hab => ?_⟩
    obtain ⟨c, cu, ac, hc⟩ : ∃ c, c ∈ u ∧ (B a ∩ B c).Nonempty ∧ δ a ≤ τ * δ c :=
      (UT uU).2.2 a hat b hbu hab
    exact ⟨c, ⟨u, uU, cu⟩, ac, hc⟩
  -- The only nontrivial bit is to check that every `a ∈ t` intersects an element `b ∈ u` with
  -- comparatively large `δ b`. Assume this is not the case, then we will contradict the maximality.
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    ⊢ Exists fun u => And (HasSubset.Subset u t) (And (u.PairwiseDisjoint B) (∀ (a …
  -/
  refine ⟨u, hu.prop.1, hu.prop.2.1, fun a hat => ?_⟩
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    a : ι
    hat : Membership.mem t a
    ⊢ Exists fun b => And (Membership.mem u b) (And (Inter.inter (B a) (B b)).None …
  -/
  by_contra! hcon
  have a_disj : ∀ c ∈ u, Disjoint (B a) (B c) := by
    intro c hc
    by_contra h
    rw [not_disjoint_iff_nonempty_inter] at h
    obtain ⟨d, du, ad, hd⟩ : ∃ d, d ∈ u ∧ (B a ∩ B d).Nonempty ∧ δ a ≤ τ * δ d :=
      hu.prop.2.2 a hat c hc h
    exact lt_irrefl _ ((hcon d du ad).trans_le hd)
  -- Let `A` be all the elements of `t` which do not intersect the family `u`. It is nonempty as it
  -- contains `a`. We will pick an element `a'` of `A` with `δ a'` almost as large as possible.
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    a : ι
    hat : Membership.mem t a
    hcon : ∀ (b : ι), Membership.mem u b → (Inter.inter (B a) (B b)).Nonempty → LT …
    a_disj : ∀ (c : ι), Membership.mem u c → Disjoint (B a) (B c)
    ⊢ False
  -/
  let A := { a' | a' ∈ t ∧ ∀ c ∈ u, Disjoint (B a') (B c) }
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    a : ι
    hat : Membership.mem t a
    hcon : ∀ (b : ι), Membership.mem u b → (Inter.inter (B a) (B b)).Nonempty → LT …
    a_disj : ∀ (c : ι), Membership.mem u c → Disjoint (B a) (B c)
    A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
    ⊢ False
  -/
  have Anonempty : A.Nonempty := ⟨a, hat, a_disj⟩
  /-
    case intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    a : ι
    hat : Membership.mem t a
    hcon : ∀ (b : ι), Membership.mem u b → (Inter.inter (B a) (B b)).Nonempty → LT …
    a_disj : ∀ (c : ι), Membership.mem u c → Disjoint (B a) (B c)
    A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
    Anonempty : A.Nonempty
    ⊢ False
  -/
  let m := sSup (δ '' A)
  have bddA : BddAbove (δ '' A) := by
    refine ⟨R, fun x xA => ?_⟩
    rcases (mem_image _ _ _).1 xA with ⟨a', ha', rfl⟩
    exact δle a' ha'.1
  obtain ⟨a', a'A, ha'⟩ : ∃ a' ∈ A, m / τ ≤ δ a' := by
    have : 0 ≤ m := (δnonneg a hat).trans (le_csSup bddA (mem_image_of_mem _ ⟨hat, a_disj⟩))
    rcases eq_or_lt_of_le this with (mzero | mpos)
    · refine ⟨a, ⟨hat, a_disj⟩, ?_⟩
      simpa only [← mzero, zero_div] using δnonneg a hat
    · have I : m / τ < m := by
        rw [div_lt_iff₀ (zero_lt_one.trans hτ)]
        conv_lhs => rw [← mul_one m]
        exact (mul_lt_mul_left mpos).2 hτ
      rcases exists_lt_of_lt_csSup (Anonempty.image _) I with ⟨x, xA, hx⟩
      rcases (mem_image _ _ _).1 xA with ⟨a', ha', rfl⟩
      exact ⟨a', ha', hx.le⟩
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    a : ι
    hat : Membership.mem t a
    hcon : ∀ (b : ι), Membership.mem u b → (Inter.inter (B a) (B b)).Nonempty → LT …
    a_disj : ∀ (c : ι), Membership.mem u c → Disjoint (B a) (B c)
    A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
    Anonempty : A.Nonempty
    m : Real := SupSet.sSup (Set.image δ A)
    bddA : BddAbove (Set.image δ A)
    a' : ι
    a'A : Membership.mem A a'
    ha' : LE.le (HDiv.hDiv m τ) (δ a')
    ⊢ False
  -/
  clear hat hcon a_disj a
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
    Anonempty : A.Nonempty
    m : Real := SupSet.sSup (Set.image δ A)
    bddA : BddAbove (Set.image δ A)
    a' : ι
    a'A : Membership.mem A a'
    ha' : LE.le (HDiv.hDiv m τ) (δ a')
    ⊢ False
  -/
  have a'_ne_u : a' ∉ u := fun H => (hne _ a'A.1).ne_empty (disjoint_self.1 (a'A.2 _ H))
  -- we claim that `u ∪ {a'}` still belongs to `T`, contradicting the maximality of `u`.
  /-
    case intro.intro.intro
    α : Type u_1
    ι : Type u_2
    B : ι → Set α
    t : Set ι
    δ : ι → Real
    τ : Real
    hτ : LT.lt 1 τ
    δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
    R : Real
    δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
    hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
    T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
    u : Set ι
    hu : Maximal (fun x => Membership.mem T x) u
    A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
    Anonempty : A.Nonempty
    m : Real := SupSet.sSup (Set.image δ A)
    bddA : BddAbove (Set.image δ A)
    a' : ι
    a'A : Membership.mem A a'
    ha' : LE.le (HDiv.hDiv m τ) (δ a')
    a'_ne_u : Not (Membership.mem u a')
    ⊢ False
  -/
  refine a'_ne_u (hu.mem_of_prop_insert ⟨?_, ?_, ?_⟩)
  · -- check that `u ∪ {a'}` is made of elements of `t`.
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      ι : Type u_2
      B : ι → Set α
      t : Set ι
      δ : ι → Real
      τ : Real
      hτ : LT.lt 1 τ
      δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
      R : Real
      δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
      hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
      T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
      u : Set ι
      hu : Maximal (fun x => Membership.mem T x) u
      A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
      Anonempty : A.Nonempty
      m : Real := SupSet.sSup (Set.image δ A)
      bddA : BddAbove (Set.image δ A)
      a' : ι
      a'A : Membership.mem A a'
      ha' : LE.le (HDiv.hDiv m τ) (δ a')
      a'_ne_u : Not (Membership.mem u a')
      ⊢ HasSubset.Subset (Insert.insert a' u) t
    -/
    rw [insert_subset_iff]
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      ι : Type u_2
      B : ι → Set α
      t : Set ι
      δ : ι → Real
      τ : Real
      hτ : LT.lt 1 τ
      δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
      R : Real
      δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
      hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
      T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
      u : Set ι
      hu : Maximal (fun x => Membership.mem T x) u
      A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
      Anonempty : A.Nonempty
      m : Real := SupSet.sSup (Set.image δ A)
      bddA : BddAbove (Set.image δ A)
      a' : ι
      a'A : Membership.mem A a'
      ha' : LE.le (HDiv.hDiv m τ) (δ a')
      a'_ne_u : Not (Membership.mem u a')
      ⊢ And (Membership.mem t a') (HasSubset.Subset u t)
    -/
    exact ⟨a'A.1, hu.prop.1⟩
    /-
      🎉 no goals
    -/
  · -- Check that `u ∪ {a'}` is a disjoint family. This follows from the fact that `a'` does not
    -- intersect `u`.
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      ι : Type u_2
      B : ι → Set α
      t : Set ι
      δ : ι → Real
      τ : Real
      hτ : LT.lt 1 τ
      δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
      R : Real
      δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
      hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
      T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
      u : Set ι
      hu : Maximal (fun x => Membership.mem T x) u
      A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
      Anonempty : A.Nonempty
      m : Real := SupSet.sSup (Set.image δ A)
      bddA : BddAbove (Set.image δ A)
      a' : ι
      a'A : Membership.mem A a'
      ha' : LE.le (HDiv.hDiv m τ) (δ a')
      a'_ne_u : Not (Membership.mem u a')
      ⊢ (Insert.insert a' u).PairwiseDisjoint B
    -/
    exact hu.prop.2.1.insert fun b bu _ => a'A.2 b bu
    /-
      🎉 no goals
    -/
  · -- check that every element `c` of `t` intersecting `u ∪ {a'}` intersects an element of this
    -- family with large `δ`.
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      ι : Type u_2
      B : ι → Set α
      t : Set ι
      δ : ι → Real
      τ : Real
      hτ : LT.lt 1 τ
      δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
      R : Real
      δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
      hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
      T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
      u : Set ι
      hu : Maximal (fun x => Membership.mem T x) u
      A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
      Anonempty : A.Nonempty
      m : Real := SupSet.sSup (Set.image δ A)
      bddA : BddAbove (Set.image δ A)
      a' : ι
      a'A : Membership.mem A a'
      ha' : LE.le (HDiv.hDiv m τ) (δ a')
      a'_ne_u : Not (Membership.mem u a')
      ⊢ ∀ (a : ι), Membership.mem t a → ∀ (b : ι), Membership.mem (Insert.insert a'  …
    -/
    intro c ct b ba'u hcb
    -- if `c` already intersects an element of `u`, then it intersects an element of `u` with
    -- large `δ` by the assumption on `u`, and there is nothing left to do.
    /-
      case intro.intro.intro.refine_3
      α : Type u_1
      ι : Type u_2
      B : ι → Set α
      t : Set ι
      δ : ι → Real
      τ : Real
      hτ : LT.lt 1 τ
      δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
      R : Real
      δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
      hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
      T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
      u : Set ι
      hu : Maximal (fun x => Membership.mem T x) u
      A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
      Anonempty : A.Nonempty
      m : Real := SupSet.sSup (Set.image δ A)
      bddA : BddAbove (Set.image δ A)
      a' : ι
      a'A : Membership.mem A a'
      ha' : LE.le (HDiv.hDiv m τ) (δ a')
      a'_ne_u : Not (Membership.mem u a')
      c : ι
      ct : Membership.mem t c
      b : ι
      ba'u : Membership.mem (Insert.insert a' u) b
      hcb : (Inter.inter (B c) (B b)).Nonempty
      ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
    -/
    by_cases H : ∃ d ∈ u, (B c ∩ B d).Nonempty
      /-
        case pos
        α : Type u_1
        ι : Type u_2
        B : ι → Set α
        t : Set ι
        δ : ι → Real
        τ : Real
        hτ : LT.lt 1 τ
        δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
        R : Real
        δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
        hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
        T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
        u : Set ι
        hu : Maximal (fun x => Membership.mem T x) u
        A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
        Anonempty : A.Nonempty
        m : Real := SupSet.sSup (Set.image δ A)
        bddA : BddAbove (Set.image δ A)
        a' : ι
        a'A : Membership.mem A a'
        ha' : LE.le (HDiv.hDiv m τ) (δ a')
        a'_ne_u : Not (Membership.mem u a')
        c : ι
        ct : Membership.mem t c
        b : ι
        ba'u : Membership.mem (Insert.insert a' u) b
        hcb : (Inter.inter (B c) (B b)).Nonempty
        H : Exists fun d => And (Membership.mem u d) (Inter.inter (B c) (B d)).Nonempty
        ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
      -/
    · rcases H with ⟨d, du, hd⟩
      /-
        case pos.intro.intro
        α : Type u_1
        ι : Type u_2
        B : ι → Set α
        t : Set ι
        δ : ι → Real
        τ : Real
        hτ : LT.lt 1 τ
        δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
        R : Real
        δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
        hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
        T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
        u : Set ι
        hu : Maximal (fun x => Membership.mem T x) u
        A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
        Anonempty : A.Nonempty
        m : Real := SupSet.sSup (Set.image δ A)
        bddA : BddAbove (Set.image δ A)
        a' : ι
        a'A : Membership.mem A a'
        ha' : LE.le (HDiv.hDiv m τ) (δ a')
        a'_ne_u : Not (Membership.mem u a')
        c : ι
        ct : Membership.mem t c
        b : ι
        ba'u : Membership.mem (Insert.insert a' u) b
        hcb : (Inter.inter (B c) (B b)).Nonempty
        d : ι
        du : Membership.mem u d
        hd : (Inter.inter (B c) (B d)).Nonempty
        ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
      -/
      rcases hu.prop.2.2 c ct d du hd with ⟨d', d'u, hd'⟩
      /-
        case pos.intro.intro.intro.intro
        α : Type u_1
        ι : Type u_2
        B : ι → Set α
        t : Set ι
        δ : ι → Real
        τ : Real
        hτ : LT.lt 1 τ
        δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
        R : Real
        δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
        hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
        T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
        u : Set ι
        hu : Maximal (fun x => Membership.mem T x) u
        A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
        Anonempty : A.Nonempty
        m : Real := SupSet.sSup (Set.image δ A)
        bddA : BddAbove (Set.image δ A)
        a' : ι
        a'A : Membership.mem A a'
        ha' : LE.le (HDiv.hDiv m τ) (δ a')
        a'_ne_u : Not (Membership.mem u a')
        c : ι
        ct : Membership.mem t c
        b : ι
        ba'u : Membership.mem (Insert.insert a' u) b
        hcb : (Inter.inter (B c) (B b)).Nonempty
        d : ι
        du : Membership.mem u d
        hd : (Inter.inter (B c) (B d)).Nonempty
        d' : ι
        d'u : Membership.mem u d'
        hd' : And (Inter.inter (B c) (B d')).Nonempty (LE.le (δ c) (HMul.hMul τ (δ d')))
        ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
      -/
      exact ⟨d', mem_insert_of_mem _ d'u, hd'⟩
      /-
        🎉 no goals
      -/
    · -- Otherwise, `c` belongs to `A`. The element of `u ∪ {a'}` that it intersects has to be `a'`.
      -- Moreover, `δ c` is smaller than the maximum `m` of `δ` over `A`, which is `≤ δ a' / τ`
      -- thanks to the good choice of `a'`. This is the desired inequality.
      /-
        case neg
        α : Type u_1
        ι : Type u_2
        B : ι → Set α
        t : Set ι
        δ : ι → Real
        τ : Real
        hτ : LT.lt 1 τ
        δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
        R : Real
        δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
        hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
        T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
        u : Set ι
        hu : Maximal (fun x => Membership.mem T x) u
        A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
        Anonempty : A.Nonempty
        m : Real := SupSet.sSup (Set.image δ A)
        bddA : BddAbove (Set.image δ A)
        a' : ι
        a'A : Membership.mem A a'
        ha' : LE.le (HDiv.hDiv m τ) (δ a')
        a'_ne_u : Not (Membership.mem u a')
        c : ι
        ct : Membership.mem t c
        b : ι
        ba'u : Membership.mem (Insert.insert a' u) b
        hcb : (Inter.inter (B c) (B b)).Nonempty
        H : Not (Exists fun d => And (Membership.mem u d) (Inter.inter (B c) (B d)).No …
        ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
      -/
      push_neg at H
      /-
        case neg
        α : Type u_1
        ι : Type u_2
        B : ι → Set α
        t : Set ι
        δ : ι → Real
        τ : Real
        hτ : LT.lt 1 τ
        δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
        R : Real
        δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
        hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
        T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
        u : Set ι
        hu : Maximal (fun x => Membership.mem T x) u
        A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
        Anonempty : A.Nonempty
        m : Real := SupSet.sSup (Set.image δ A)
        bddA : BddAbove (Set.image δ A)
        a' : ι
        a'A : Membership.mem A a'
        ha' : LE.le (HDiv.hDiv m τ) (δ a')
        a'_ne_u : Not (Membership.mem u a')
        c : ι
        ct : Membership.mem t c
        b : ι
        ba'u : Membership.mem (Insert.insert a' u) b
        hcb : (Inter.inter (B c) (B b)).Nonempty
        H : ∀ (d : ι), Membership.mem u d → Eq (Inter.inter (B c) (B d)) EmptyCollecti …
        ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
      -/
      simp only [← disjoint_iff_inter_eq_empty] at H
      /-
        case neg
        α : Type u_1
        ι : Type u_2
        B : ι → Set α
        t : Set ι
        δ : ι → Real
        τ : Real
        hτ : LT.lt 1 τ
        δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
        R : Real
        δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
        hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
        T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
        u : Set ι
        hu : Maximal (fun x => Membership.mem T x) u
        A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
        Anonempty : A.Nonempty
        m : Real := SupSet.sSup (Set.image δ A)
        bddA : BddAbove (Set.image δ A)
        a' : ι
        a'A : Membership.mem A a'
        ha' : LE.le (HDiv.hDiv m τ) (δ a')
        a'_ne_u : Not (Membership.mem u a')
        c : ι
        ct : Membership.mem t c
        b : ι
        ba'u : Membership.mem (Insert.insert a' u) b
        hcb : (Inter.inter (B c) (B b)).Nonempty
        H : ∀ (d : ι), Membership.mem u d → Disjoint (B c) (B d)
        ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
      -/
      rcases mem_insert_iff.1 ba'u with (rfl | H')
        /-
          case neg.inl
          α : Type u_1
          ι : Type u_2
          B : ι → Set α
          t : Set ι
          δ : ι → Real
          τ : Real
          hτ : LT.lt 1 τ
          δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
          R : Real
          δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
          hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
          T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
          u : Set ι
          hu : Maximal (fun x => Membership.mem T x) u
          A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
          Anonempty : A.Nonempty
          m : Real := SupSet.sSup (Set.image δ A)
          bddA : BddAbove (Set.image δ A)
          c : ι
          ct : Membership.mem t c
          b : ι
          hcb : (Inter.inter (B c) (B b)).Nonempty
          H : ∀ (d : ι), Membership.mem u d → Disjoint (B c) (B d)
          a'A : Membership.mem A b
          ha' : LE.le (HDiv.hDiv m τ) (δ b)
          a'_ne_u : Not (Membership.mem u b)
          ba'u : Membership.mem (Insert.insert b u) b
          ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert b u) c_1) (And (Inter.i …
        -/
      · refine ⟨b, mem_insert _ _, hcb, ?_⟩
        calc
          δ c ≤ m := le_csSup bddA (mem_image_of_mem _ ⟨ct, H⟩)
          _ = τ * (m / τ) := by field_simp [(zero_lt_one.trans hτ).ne']
          _ ≤ τ * δ b := by gcongr
        /-
          case neg.inr
          α : Type u_1
          ι : Type u_2
          B : ι → Set α
          t : Set ι
          δ : ι → Real
          τ : Real
          hτ : LT.lt 1 τ
          δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
          R : Real
          δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
          hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
          T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
          u : Set ι
          hu : Maximal (fun x => Membership.mem T x) u
          A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
          Anonempty : A.Nonempty
          m : Real := SupSet.sSup (Set.image δ A)
          bddA : BddAbove (Set.image δ A)
          a' : ι
          a'A : Membership.mem A a'
          ha' : LE.le (HDiv.hDiv m τ) (δ a')
          a'_ne_u : Not (Membership.mem u a')
          c : ι
          ct : Membership.mem t c
          b : ι
          ba'u : Membership.mem (Insert.insert a' u) b
          hcb : (Inter.inter (B c) (B b)).Nonempty
          H : ∀ (d : ι), Membership.mem u d → Disjoint (B c) (B d)
          H' : Membership.mem u b
          ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
        -/
      · rw [← not_disjoint_iff_nonempty_inter] at hcb
        /-
          case neg.inr
          α : Type u_1
          ι : Type u_2
          B : ι → Set α
          t : Set ι
          δ : ι → Real
          τ : Real
          hτ : LT.lt 1 τ
          δnonneg : ∀ (a : ι), Membership.mem t a → LE.le 0 (δ a)
          R : Real
          δle : ∀ (a : ι), Membership.mem t a → LE.le (δ a) R
          hne : ∀ (a : ι), Membership.mem t a → (B a).Nonempty
          T : Set (Set ι) := setOf fun u => And (HasSubset.Subset u t) (And (u.PairwiseD …
          u : Set ι
          hu : Maximal (fun x => Membership.mem T x) u
          A : Set ι := setOf fun a' => And (Membership.mem t a') (∀ (c : ι), Membership. …
          Anonempty : A.Nonempty
          m : Real := SupSet.sSup (Set.image δ A)
          bddA : BddAbove (Set.image δ A)
          a' : ι
          a'A : Membership.mem A a'
          ha' : LE.le (HDiv.hDiv m τ) (δ a')
          a'_ne_u : Not (Membership.mem u a')
          c : ι
          ct : Membership.mem t c
          b : ι
          ba'u : Membership.mem (Insert.insert a' u) b
          hcb : Not (Disjoint (B c) (B b))
          H : ∀ (d : ι), Membership.mem u d → Disjoint (B c) (B d)
          H' : Membership.mem u b
          ⊢ Exists fun c_1 => And (Membership.mem (Insert.insert a' u) c_1) (And (Inter. …
        -/
        exact (hcb (H _ H')).elim
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-12-25")]
alias exists_disjoint_subfamily_covering_enlargment :=
  exists_disjoint_subfamily_covering_enlargement


/-- Vitali covering theorem, closed balls version: given a family `t` of closed balls, one can
extract a disjoint subfamily `u ⊆ t` so that all balls in `t` are covered by the τ-times
dilations of balls in `u`, for some `τ > 3`. -/
theorem exists_disjoint_subfamily_covering_enlargement_closedBall
    [PseudoMetricSpace α] (t : Set ι)
    (x : ι → α) (r : ι → ℝ) (R : ℝ) (hr : ∀ a ∈ t, r a ≤ R) (τ : ℝ) (hτ : 3 < τ) :
    ∃ u ⊆ t,
      (u.PairwiseDisjoint fun a => closedBall (x a) (r a)) ∧
        ∀ a ∈ t, ∃ b ∈ u, closedBall (x a) (r a) ⊆ closedBall (x b) (τ * r b) := by
  /-
    α : Type u_1
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    t : Set ι
    x : ι → α
    r : ι → Real
    R : Real
    hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
    τ : Real
    hτ : LT.lt 3 τ
    ⊢ Exists fun u => And (HasSubset.Subset u t) (And (u.PairwiseDisjoint fun a => …
  -/
  rcases eq_empty_or_nonempty t with (rfl | _)
    /-
      case inl
      α : Type u_1
      ι : Type u_2
      inst✝ : PseudoMetricSpace α
      x : ι → α
      r : ι → Real
      R τ : Real
      hτ : LT.lt 3 τ
      hr : ∀ (a : ι), Membership.mem EmptyCollection.emptyCollection a → LE.le (r a) R
      ⊢ Exists fun u => And (HasSubset.Subset u EmptyCollection.emptyCollection) (An …
    -/
  · exact ⟨∅, Subset.refl _, pairwiseDisjoint_empty, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    t : Set ι
    x : ι → α
    r : ι → Real
    R : Real
    hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
    τ : Real
    hτ : LT.lt 3 τ
    h✝ : t.Nonempty
    ⊢ Exists fun u => And (HasSubset.Subset u t) (And (u.PairwiseDisjoint fun a => …
  -/
  by_cases ht : ∀ a ∈ t, r a < 0
  · exact ⟨t, Subset.rfl, fun a ha b _ _ => by
      #adaptation_note /-- nightly-2024-03-16
      Previously `Function.onFun` unfolded in the following `simp only`,
      but now needs a separate `rw`.
      This may be a bug: a no import minimization may be required. -/
      rw [Function.onFun]
      simp only [Function.onFun, closedBall_eq_empty.2 (ht a ha), empty_disjoint],
      fun a ha => ⟨a, ha, by simp only [closedBall_eq_empty.2 (ht a ha), empty_subset]⟩⟩
  /-
    case neg
    α : Type u_1
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    t : Set ι
    x : ι → α
    r : ι → Real
    R : Real
    hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
    τ : Real
    hτ : LT.lt 3 τ
    h✝ : t.Nonempty
    ht : Not (∀ (a : ι), Membership.mem t a → LT.lt (r a) 0)
    ⊢ Exists fun u => And (HasSubset.Subset u t) (And (u.PairwiseDisjoint fun a => …
  -/
  push_neg at ht
  /-
    case neg
    α : Type u_1
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    t : Set ι
    x : ι → α
    r : ι → Real
    R : Real
    hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
    τ : Real
    hτ : LT.lt 3 τ
    h✝ : t.Nonempty
    ht : Exists fun a => And (Membership.mem t a) (LE.le 0 (r a))
    ⊢ Exists fun u => And (HasSubset.Subset u t) (And (u.PairwiseDisjoint fun a => …
  -/
  let t' := { a ∈ t | 0 ≤ r a }
  rcases exists_disjoint_subfamily_covering_enlargement (fun a => closedBall (x a) (r a)) t' r
      ((τ - 1) / 2) (by linarith) (fun a ha => ha.2) R (fun a ha => hr a ha.1) fun a ha =>
      ⟨x a, mem_closedBall_self ha.2⟩ with
    ⟨u, ut', u_disj, hu⟩
  have A : ∀ a ∈ t', ∃ b ∈ u, closedBall (x a) (r a) ⊆ closedBall (x b) (τ * r b) := by
    intro a ha
    rcases hu a ha with ⟨b, bu, hb, rb⟩
    refine ⟨b, bu, ?_⟩
    have : dist (x a) (x b) ≤ r a + r b := dist_le_add_of_nonempty_closedBall_inter_closedBall hb
    apply closedBall_subset_closedBall'
    linarith
  /-
    case neg.intro.intro.intro
    α : Type u_1
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    t : Set ι
    x : ι → α
    r : ι → Real
    R : Real
    hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
    τ : Real
    hτ : LT.lt 3 τ
    h✝ : t.Nonempty
    ht : Exists fun a => And (Membership.mem t a) (LE.le 0 (r a))
    t' : Set ι := setOf fun a => And (Membership.mem t a) (LE.le 0 (r a))
    u : Set ι
    ut' : HasSubset.Subset u t'
    u_disj : u.PairwiseDisjoint fun a => Metric.closedBall (x a) (r a)
    hu : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b) …
    A : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b)  …
    ⊢ Exists fun u => And (HasSubset.Subset u t) (And (u.PairwiseDisjoint fun a => …
  -/
  refine ⟨u, ut'.trans fun a ha => ha.1, u_disj, fun a ha => ?_⟩
  /-
    case neg.intro.intro.intro
    α : Type u_1
    ι : Type u_2
    inst✝ : PseudoMetricSpace α
    t : Set ι
    x : ι → α
    r : ι → Real
    R : Real
    hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
    τ : Real
    hτ : LT.lt 3 τ
    h✝ : t.Nonempty
    ht : Exists fun a => And (Membership.mem t a) (LE.le 0 (r a))
    t' : Set ι := setOf fun a => And (Membership.mem t a) (LE.le 0 (r a))
    u : Set ι
    ut' : HasSubset.Subset u t'
    u_disj : u.PairwiseDisjoint fun a => Metric.closedBall (x a) (r a)
    hu : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b) …
    A : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b)  …
    a : ι
    ha : Membership.mem t a
    ⊢ Exists fun b => And (Membership.mem u b) (HasSubset.Subset (Metric.closedBal …
  -/
  rcases le_or_lt 0 (r a) with (h'a | h'a)
    /-
      case neg.intro.intro.intro.inl
      α : Type u_1
      ι : Type u_2
      inst✝ : PseudoMetricSpace α
      t : Set ι
      x : ι → α
      r : ι → Real
      R : Real
      hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
      τ : Real
      hτ : LT.lt 3 τ
      h✝ : t.Nonempty
      ht : Exists fun a => And (Membership.mem t a) (LE.le 0 (r a))
      t' : Set ι := setOf fun a => And (Membership.mem t a) (LE.le 0 (r a))
      u : Set ι
      ut' : HasSubset.Subset u t'
      u_disj : u.PairwiseDisjoint fun a => Metric.closedBall (x a) (r a)
      hu : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b) …
      A : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b)  …
      a : ι
      ha : Membership.mem t a
      h'a : LE.le 0 (r a)
      ⊢ Exists fun b => And (Membership.mem u b) (HasSubset.Subset (Metric.closedBal …
    -/
  · exact A a ⟨ha, h'a⟩
    /-
      🎉 no goals
    -/
    /-
      case neg.intro.intro.intro.inr
      α : Type u_1
      ι : Type u_2
      inst✝ : PseudoMetricSpace α
      t : Set ι
      x : ι → α
      r : ι → Real
      R : Real
      hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
      τ : Real
      hτ : LT.lt 3 τ
      h✝ : t.Nonempty
      ht : Exists fun a => And (Membership.mem t a) (LE.le 0 (r a))
      t' : Set ι := setOf fun a => And (Membership.mem t a) (LE.le 0 (r a))
      u : Set ι
      ut' : HasSubset.Subset u t'
      u_disj : u.PairwiseDisjoint fun a => Metric.closedBall (x a) (r a)
      hu : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b) …
      A : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b)  …
      a : ι
      ha : Membership.mem t a
      h'a : LT.lt (r a) 0
      ⊢ Exists fun b => And (Membership.mem u b) (HasSubset.Subset (Metric.closedBal …
    -/
  · rcases ht with ⟨b, rb⟩
    /-
      case neg.intro.intro.intro.inr.intro
      α : Type u_1
      ι : Type u_2
      inst✝ : PseudoMetricSpace α
      t : Set ι
      x : ι → α
      r : ι → Real
      R : Real
      hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
      τ : Real
      hτ : LT.lt 3 τ
      h✝ : t.Nonempty
      t' : Set ι := setOf fun a => And (Membership.mem t a) (LE.le 0 (r a))
      u : Set ι
      ut' : HasSubset.Subset u t'
      u_disj : u.PairwiseDisjoint fun a => Metric.closedBall (x a) (r a)
      hu : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b) …
      A : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b)  …
      a : ι
      ha : Membership.mem t a
      h'a : LT.lt (r a) 0
      b : ι
      rb : And (Membership.mem t b) (LE.le 0 (r b))
      ⊢ Exists fun b => And (Membership.mem u b) (HasSubset.Subset (Metric.closedBal …
    -/
    rcases A b ⟨rb.1, rb.2⟩ with ⟨c, cu, _⟩
    /-
      case neg.intro.intro.intro.inr.intro.intro.intro
      α : Type u_1
      ι : Type u_2
      inst✝ : PseudoMetricSpace α
      t : Set ι
      x : ι → α
      r : ι → Real
      R : Real
      hr : ∀ (a : ι), Membership.mem t a → LE.le (r a) R
      τ : Real
      hτ : LT.lt 3 τ
      h✝ : t.Nonempty
      t' : Set ι := setOf fun a => And (Membership.mem t a) (LE.le 0 (r a))
      u : Set ι
      ut' : HasSubset.Subset u t'
      u_disj : u.PairwiseDisjoint fun a => Metric.closedBall (x a) (r a)
      hu : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b) …
      A : ∀ (a : ι), Membership.mem t' a → Exists fun b => And (Membership.mem u b)  …
      a : ι
      ha : Membership.mem t a
      h'a : LT.lt (r a) 0
      b : ι
      rb : And (Membership.mem t b) (LE.le 0 (r b))
      c : ι
      cu : Membership.mem u c
      right✝ : HasSubset.Subset (Metric.closedBall (x b) (r b)) (Metric.closedBall ( …
      ⊢ Exists fun b => And (Membership.mem u b) (HasSubset.Subset (Metric.closedBal …
    -/
    exact ⟨c, cu, by simp only [closedBall_eq_empty.2 h'a, empty_subset]⟩
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-25")]
alias exists_disjoint_subfamily_covering_enlargment_closedBall :=
  exists_disjoint_subfamily_covering_enlargement_closedBall


/-- The measurable Vitali covering theorem. Assume one is given a family `t` of closed sets with
nonempty interior, such that each `a ∈ t` is included in a ball `B (x, r)` and covers a definite
proportion of the ball `B (x, 3 r)` for a given measure `μ` (think of the situation where `μ` is
a doubling measure and `t` is a family of balls). Consider a (possibly non-measurable) set `s`
at which the family is fine, i.e., every point of `s` belongs to arbitrarily small elements of `t`.
Then one can extract from `t` a disjoint subfamily that covers almost all `s`.

For more flexibility, we give a statement with a parameterized family of sets.
-/
theorem exists_disjoint_covering_ae
    [PseudoMetricSpace α] [MeasurableSpace α] [OpensMeasurableSpace α]
    [SecondCountableTopology α] (μ : Measure α) [IsLocallyFiniteMeasure μ] (s : Set α) (t : Set ι)
    (C : ℝ≥0) (r : ι → ℝ) (c : ι → α) (B : ι → Set α) (hB : ∀ a ∈ t, B a ⊆ closedBall (c a) (r a))
    (μB : ∀ a ∈ t, μ (closedBall (c a) (3 * r a)) ≤ C * μ (B a))
    (ht : ∀ a ∈ t, (interior (B a)).Nonempty) (h't : ∀ a ∈ t, IsClosed (B a))
    (hf : ∀ x ∈ s, ∀ ε > (0 : ℝ), ∃ a ∈ t, r a ≤ ε ∧ c a = x) :
    ∃ u ⊆ t, u.Countable ∧ u.PairwiseDisjoint B ∧ μ (s \ ⋃ a ∈ u, B a) = 0 := by
  /- The idea of the proof is the following. Assume for simplicity that `μ` is finite. Applying the
  abstract Vitali covering theorem with `δ = r` given by `hf`, one obtains a disjoint subfamily `u`,
  such that any element of `t` intersects an element of `u` with comparable radius. Fix `ε > 0`.
  Since the elements of `u` have summable measure, one can remove finitely elements `w_1, ..., w_n`.
  so that the measure of the remaining elements is `< ε`. Consider now a point `z` not
  in the `w_i`. There is a small ball around `z` not intersecting the `w_i` (as they are closed),
  an element `a ∈ t` contained in this small ball (as the family `t` is fine at `z`) and an element
  `b ∈ u` intersecting `a`, with comparable radius (by definition of `u`). Then `z` belongs to the
  enlargement of `b`. This shows that `s \ (w_1 ∪ ... ∪ w_n)` is contained in
  `⋃ (b ∈ u \ {w_1, ... w_n}) (enlargement of b)`. The measure of the latter set is bounded by
  `∑ (b ∈ u \ {w_1, ... w_n}) C * μ b` (by the doubling property of the measure), which is at most
  `C ε`. Letting `ε` tend to `0` shows that `s` is almost everywhere covered by the family `u`.

  For the real argument, the measure is only locally finite. Therefore, we implement the same
  strategy, but locally restricted to balls on which the measure is finite. For this, we do not
  use the whole family `t`, but a subfamily `t'` supported on small balls (which is possible since
  the family is assumed to be fine at every point of `s`).
  -/
  classical
  -- choose around each `x` a small ball on which the measure is finite
  have : ∀ x, ∃ R, 0 < R ∧ R ≤ 1 ∧ μ (closedBall x (20 * R)) < ∞ := fun x ↦ by
    refine ((eventually_le_nhds one_pos).and ?_).exists_gt
    refine (tendsto_closedBall_smallSets x).comp ?_ (μ.finiteAt_nhds x).eventually
    exact Continuous.tendsto' (by fun_prop) _ _ (mul_zero _)
  choose R hR0 hR1 hRμ using this
  -- we restrict to a subfamily `t'` of `t`, made of elements small enough to ensure that
  -- they only see a finite part of the measure, and with a doubling property
  let t' := { a ∈ t | r a ≤ R (c a) }
  -- extract a disjoint subfamily `u` of `t'` thanks to the abstract Vitali covering theorem.
  obtain ⟨u, ut', u_disj, hu⟩ : ∃ u ⊆ t',
      u.PairwiseDisjoint B ∧ ∀ a ∈ t', ∃ b ∈ u, (B a ∩ B b).Nonempty ∧ r a ≤ 2 * r b := by
    have A : ∀ a ∈ t', r a ≤ 1 := by
      intro a ha
      apply ha.2.trans (hR1 (c a))
    have A' : ∀ a ∈ t', (B a).Nonempty :=
      fun a hat' => Set.Nonempty.mono interior_subset (ht a hat'.1)
    refine exists_disjoint_subfamily_covering_enlargement
      B t' r 2 one_lt_two (fun a ha => ?_) 1 A A'
    exact nonempty_closedBall.1 ((A' a ha).mono (hB a ha.1))
  have ut : u ⊆ t := fun a hau => (ut' hau).1
  -- As the space is second countable, the family is countable since all its sets have nonempty
  -- interior.
  have u_count : u.Countable := u_disj.countable_of_nonempty_interior fun a ha => ht a (ut ha)
  -- the family `u` will be the desired family
  refine ⟨u, fun a hat' => (ut' hat').1, u_count, u_disj, ?_⟩
  -- it suffices to show that it covers almost all `s` locally around each point `x`.
  refine measure_null_of_locally_null _ fun x _ => ?_
  -- let `v` be the subfamily of `u` made of those sets intersecting the small ball `ball x (r x)`
  let v := { a ∈ u | (B a ∩ ball x (R x)).Nonempty }
  have vu : v ⊆ u := fun a ha => ha.1
  -- they are all contained in a fixed ball of finite measure, thanks to our choice of `t'`
  obtain ⟨K, μK, hK⟩ : ∃ K, μ (closedBall x K) < ∞ ∧
      ∀ a ∈ u, (B a ∩ ball x (R x)).Nonempty → B a ⊆ closedBall x K := by
    have Idist_v : ∀ a ∈ v, dist (c a) x ≤ r a + R x := by
      intro a hav
      apply dist_le_add_of_nonempty_closedBall_inter_closedBall
      refine hav.2.mono ?_
      apply inter_subset_inter _ ball_subset_closedBall
      exact hB a (ut (vu hav))
    set R0 := sSup (r '' v) with R0_def
    have R0_bdd : BddAbove (r '' v) := by
      refine ⟨1, fun r' hr' => ?_⟩
      rcases (mem_image _ _ _).1 hr' with ⟨b, hb, rfl⟩
      exact le_trans (ut' (vu hb)).2 (hR1 (c b))
    rcases le_total R0 (R x) with (H | H)
    · refine ⟨20 * R x, hRμ x, fun a au hax => ?_⟩
      refine (hB a (ut au)).trans ?_
      apply closedBall_subset_closedBall'
      have : r a ≤ R0 := le_csSup R0_bdd (mem_image_of_mem _ ⟨au, hax⟩)
      linarith [Idist_v a ⟨au, hax⟩, hR0 x]
    · have R0pos : 0 < R0 := (hR0 x).trans_le H
      have vnonempty : v.Nonempty := by
        by_contra h
        rw [nonempty_iff_ne_empty, Classical.not_not] at h
        rw [h, image_empty, Real.sSup_empty] at R0_def
        exact lt_irrefl _ (R0pos.trans_le (le_of_eq R0_def))
      obtain ⟨a, hav, R0a⟩ : ∃ a ∈ v, R0 / 2 < r a := by
        obtain ⟨r', r'mem, hr'⟩ : ∃ r' ∈ r '' v, R0 / 2 < r' :=
          exists_lt_of_lt_csSup (vnonempty.image _) (half_lt_self R0pos)
        rcases (mem_image _ _ _).1 r'mem with ⟨a, hav, rfl⟩
        exact ⟨a, hav, hr'⟩
      refine ⟨8 * R0, ?_, ?_⟩
      · apply lt_of_le_of_lt (measure_mono _) (hRμ (c a))
        apply closedBall_subset_closedBall'
        rw [dist_comm]
        linarith [Idist_v a hav, (ut' (vu hav)).2]
      · intro b bu hbx
        refine (hB b (ut bu)).trans ?_
        apply closedBall_subset_closedBall'
        have : r b ≤ R0 := le_csSup R0_bdd (mem_image_of_mem _ ⟨bu, hbx⟩)
        linarith [Idist_v b ⟨bu, hbx⟩]
  -- we will show that, in `ball x (R x)`, almost all `s` is covered by the family `u`.
  refine ⟨_ ∩ ball x (R x), inter_mem_nhdsWithin _ (ball_mem_nhds _ (hR0 _)),
    nonpos_iff_eq_zero.mp (le_of_forall_le_of_dense fun ε εpos => ?_)⟩
  -- the elements of `v` are disjoint and all contained in a finite volume ball, hence the sum
  -- of their measures is finite.
  have I : (∑' a : v, μ (B a)) < ∞ := by
    calc
      (∑' a : v, μ (B a)) = μ (⋃ a ∈ v, B a) := by
        rw [measure_biUnion (u_count.mono vu) _ fun a ha => (h't _ (vu.trans ut ha)).measurableSet]
        exact u_disj.subset vu
      _ ≤ μ (closedBall x K) := (measure_mono (iUnion₂_subset fun a ha => hK a (vu ha) ha.2))
      _ < ∞ := μK
  -- we can obtain a finite subfamily of `v`, such that the measures of the remaining elements
  -- add up to an arbitrarily small number, say `ε / C`.
  obtain ⟨w, hw⟩ : ∃ w : Finset v, (∑' a : { a // a ∉ w }, μ (B a)) < ε / C :=
    haveI : 0 < ε / C := by
      simp only [ENNReal.div_pos_iff, εpos.ne', ENNReal.coe_ne_top, Ne, not_false_iff,
        and_self_iff]
    ((tendsto_order.1 (ENNReal.tendsto_tsum_compl_atTop_zero I.ne)).2 _ this).exists
  -- main property: the points `z` of `s` which are not covered by `u` are contained in the
  -- enlargements of the elements not in `w`.
  have M : (s \ ⋃ a ∈ u, B a) ∩
      ball x (R x) ⊆ ⋃ a : { a // a ∉ w }, closedBall (c a) (3 * r a) := by
    intro z hz
    set k := ⋃ (a : v) (_ : a ∈ w), B a
    have k_closed : IsClosed k := isClosed_biUnion_finset fun i _ => h't _ (ut (vu i.2))
    have z_notmem_k : z ∉ k := by
      simp only [k, not_exists, exists_prop, mem_iUnion, mem_sep_iff, forall_exists_index,
        SetCoe.exists, not_and, exists_and_right, Subtype.coe_mk]
      intro b hbv _ h'z
      have : z ∈ (s \ ⋃ a ∈ u, B a) ∩ ⋃ a ∈ u, B a :=
        mem_inter (mem_of_mem_inter_left hz) (mem_biUnion (vu hbv) h'z)
      simpa only [diff_inter_self]
    -- since the elements of `w` are closed and finitely many, one can find a small ball around `z`
    -- not intersecting them
    have : ball x (R x) \ k ∈ 𝓝 z := by
      apply IsOpen.mem_nhds (isOpen_ball.sdiff k_closed) _
      exact (mem_diff _).2 ⟨mem_of_mem_inter_right hz, z_notmem_k⟩
    obtain ⟨d, dpos, hd⟩ : ∃ d, 0 < d ∧ closedBall z d ⊆ ball x (R x) \ k :=
      nhds_basis_closedBall.mem_iff.1 this
    -- choose an element `a` of the family `t` contained in this small ball
    obtain ⟨a, hat, ad, rfl⟩ : ∃ a ∈ t, r a ≤ min d (R z) ∧ c a = z :=
      hf z ((mem_diff _).1 (mem_of_mem_inter_left hz)).1 (min d (R z)) (lt_min dpos (hR0 z))
    have ax : B a ⊆ ball x (R x) := by
      refine (hB a hat).trans ?_
      refine Subset.trans ?_ (hd.trans Set.diff_subset)
      exact closedBall_subset_closedBall (ad.trans (min_le_left _ _))
    -- it intersects an element `b` of `u` with comparable diameter, by definition of `u`
    obtain ⟨b, bu, ab, bdiam⟩ : ∃ b ∈ u, (B a ∩ B b).Nonempty ∧ r a ≤ 2 * r b :=
      hu a ⟨hat, ad.trans (min_le_right _ _)⟩
    have bv : b ∈ v := by
      refine ⟨bu, ab.mono ?_⟩
      rw [inter_comm]
      exact inter_subset_inter_right _ ax
    let b' : v := ⟨b, bv⟩
    -- `b` cannot belong to `w`, as the elements of `w` do not intersect `closedBall z d`,
    -- contrary to `b`
    have b'_notmem_w : b' ∉ w := by
      intro b'w
      have b'k : B b' ⊆ k := @Finset.subset_set_biUnion_of_mem _ _ _ (fun y : v => B y) _ b'w
      have : (ball x (R x) \ k ∩ k).Nonempty := by
        apply ab.mono (inter_subset_inter _ b'k)
        refine ((hB _ hat).trans ?_).trans hd
        exact closedBall_subset_closedBall (ad.trans (min_le_left _ _))
      simpa only [diff_inter_self, Set.not_nonempty_empty]
    let b'' : { a // a ∉ w } := ⟨b', b'_notmem_w⟩
    -- since `a` and `b` have comparable diameters, it follows that `z` belongs to the
    -- enlargement of `b`
    have zb : c a ∈ closedBall (c b) (3 * r b) := by
      rcases ab with ⟨e, ⟨ea, eb⟩⟩
      have A : dist (c a) e ≤ r a := mem_closedBall'.1 (hB a hat ea)
      have B : dist e (c b) ≤ r b := mem_closedBall.1 (hB b (ut bu) eb)
      simp only [mem_closedBall]
      linarith only [dist_triangle (c a) e (c b), A, B, bdiam]
    suffices H : closedBall (c b'') (3 * r b'')
        ⊆ ⋃ a : { a // a ∉ w }, closedBall (c a) (3 * r a) from H zb
    exact subset_iUnion (fun a : { a // a ∉ w } => closedBall (c a) (3 * r a)) b''
  -- now that we have proved our main inclusion, we can use it to estimate the measure of the points
  -- in `ball x (r x)` not covered by `u`.
  haveI : Countable v := (u_count.mono vu).to_subtype
  calc
    μ ((s \ ⋃ a ∈ u, B a) ∩ ball x (R x)) ≤ μ (⋃ a : { a // a ∉ w }, closedBall (c a) (3 * r a)) :=
      measure_mono M
    _ ≤ ∑' a : { a // a ∉ w }, μ (closedBall (c a) (3 * r a)) := measure_iUnion_le _
    _ ≤ ∑' a : { a // a ∉ w }, C * μ (B a) := (ENNReal.tsum_le_tsum fun a => μB a (ut (vu a.1.2)))
    _ = C * ∑' a : { a // a ∉ w }, μ (B a) := ENNReal.tsum_mul_left
    _ ≤ C * (ε / C) := by gcongr
    _ ≤ ε := ENNReal.mul_div_le


/-- Assume that around every point there are arbitrarily small scales at which the measure is
doubling. Then the set of closed sets `a` with nonempty interior contained in `closedBall x r` and
covering a fixed proportion `1/C` of the ball `closedBall x (3 * r)` forms a Vitali family.
This is essentially a restatement of the measurable Vitali theorem. -/
protected def vitaliFamily [PseudoMetricSpace α] [MeasurableSpace α] [OpensMeasurableSpace α]
    [SecondCountableTopology α] (μ : Measure α) [IsLocallyFiniteMeasure μ] (C : ℝ≥0)
    (h : ∀ x, ∃ᶠ r in 𝓝[>] 0, μ (closedBall x (3 * r)) ≤ C * μ (closedBall x r)) :
    VitaliFamily μ where
  setsAt x := { a | IsClosed a ∧ (interior a).Nonempty ∧
    ∃ r, a ⊆ closedBall x r ∧ μ (closedBall x (3 * r)) ≤ C * μ a }
  measurableSet _ _ ha := ha.1.measurableSet
  nonempty_interior _ _ ha := ha.2.1
  nontrivial x ε εpos := by
    obtain ⟨r, μr, rpos, rε⟩ :
        ∃ r, μ (closedBall x (3 * r)) ≤ C * μ (closedBall x r) ∧ r ∈ Ioc (0 : ℝ) ε :=
      ((h x).and_eventually (Ioc_mem_nhdsGT εpos)).exists
    refine
      ⟨closedBall x r, ⟨isClosed_ball, ?_, ⟨r, Subset.rfl, μr⟩⟩, closedBall_subset_closedBall rε⟩
    /-
      case intro.intro.intro
      α : Type u_1
      ι : Type u_2
      inst✝⁴ : PseudoMetricSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : SecondCountableTopology α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      C : NNReal
      h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
      x : α
      ε : Real
      εpos : GT.gt ε 0
      r : Real
      μr : LE.le (μ (Metric.closedBall x (HMul.hMul 3 r))) (HMul.hMul (↑C) (μ (Metri …
      rpos : LT.lt 0 r
      rε : LE.le r ε
      ⊢ (interior (Metric.closedBall x r)).Nonempty
    -/
    exact (nonempty_ball.2 rpos).mono ball_subset_interior_closedBall
    /-
      🎉 no goals
    -/
  covering := by
    /-
      α : Type u_1
      ι : Type u_2
      inst✝⁴ : PseudoMetricSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : SecondCountableTopology α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      C : NNReal
      h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
      ⊢ ∀ (s : Set α) (f : α → Set (Set α)), (∀ (x : α), Membership.mem s x → HasSub …
    -/
    intro s f fsubset ffine
    let t : Set (ℝ × α × Set α) :=
      { p | p.2.2 ⊆ closedBall p.2.1 p.1 ∧ μ (closedBall p.2.1 (3 * p.1)) ≤ C * μ p.2.2 ∧
            (interior p.2.2).Nonempty ∧ IsClosed p.2.2 ∧ p.2.2 ∈ f p.2.1 ∧ p.2.1 ∈ s }
    have A : ∀ x ∈ s, ∀ ε : ℝ, ε > 0 → ∃ p, p ∈ t ∧ p.1 ≤ ε ∧ p.2.1 = x := by
      intro x xs ε εpos
      rcases ffine x xs ε εpos with ⟨a, ha, h'a⟩
      rcases fsubset x xs ha with ⟨a_closed, a_int, ⟨r, ar, μr⟩⟩
      refine ⟨⟨min r ε, x, a⟩, ⟨?_, ?_, a_int, a_closed, ha, xs⟩, min_le_right _ _, rfl⟩
      · rcases min_cases r ε with (h' | h') <;> rwa [h'.1]
      · apply le_trans ?_ μr
        gcongr
        apply min_le_left
    rcases exists_disjoint_covering_ae μ s t C (fun p => p.1) (fun p => p.2.1) (fun p => p.2.2)
        (fun p hp => hp.1) (fun p hp => hp.2.1) (fun p hp => hp.2.2.1) (fun p hp => hp.2.2.2.1) A
      with ⟨t', t't, _, t'_disj, μt'⟩
    /-
      case intro.intro.intro.intro
      α : Type u_1
      ι : Type u_2
      inst✝⁴ : PseudoMetricSpace α
      inst✝³ : MeasurableSpace α
      inst✝² : OpensMeasurableSpace α
      inst✝¹ : SecondCountableTopology α
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      C : NNReal
      h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
      s : Set α
      f : α → Set (Set α)
      fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
      ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
      t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
      A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
      t' : Set (Prod Real (Prod α (Set α)))
      t't : HasSubset.Subset t' t
      left✝ : t'.Countable
      t'_disj : t'.PairwiseDisjoint fun p => p.2.2
      μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
      ⊢ Exists fun t => And (∀ (p : Prod α (Set α)), Membership.mem t p → Membership …
    -/
    refine ⟨(fun p : ℝ × α × Set α => p.2) '' t', ?_, ?_, ?_, ?_⟩
      /-
        case intro.intro.intro.intro.refine_1
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        ⊢ ∀ (p : Prod α (Set α)), Membership.mem (Set.image (fun p => p.2) t') p → Mem …
      -/
    · rintro - ⟨q, hq, rfl⟩
      /-
        case intro.intro.intro.intro.refine_1.intro.intro
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        q : Prod Real (Prod α (Set α))
        hq : Membership.mem t' q
        ⊢ Membership.mem s ((fun p => p.2) q).1
      -/
      exact (t't hq).2.2.2.2.2
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_2
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        ⊢ (Set.image (fun p => p.2) t').PairwiseDisjoint fun p => p.2
      -/
    · rintro p ⟨q, hq, rfl⟩ p' ⟨q', hq', rfl⟩ hqq'
      /-
        case intro.intro.intro.intro.refine_2.intro.intro.intro.intro
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        q : Prod Real (Prod α (Set α))
        hq : Membership.mem t' q
        q' : Prod Real (Prod α (Set α))
        hq' : Membership.mem t' q'
        hqq' : Ne ((fun p => p.2) q) ((fun p => p.2) q')
        ⊢ Function.onFun Disjoint (fun p => p.2) ((fun p => p.2) q) ((fun p => p.2) q')
      -/
      exact t'_disj hq hq' (ne_of_apply_ne _ hqq')
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_3
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        ⊢ ∀ (p : Prod α (Set α)), Membership.mem (Set.image (fun p => p.2) t') p → Mem …
      -/
    · rintro - ⟨q, hq, rfl⟩
      /-
        case intro.intro.intro.intro.refine_3.intro.intro
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        q : Prod Real (Prod α (Set α))
        hq : Membership.mem t' q
        ⊢ Membership.mem (f ((fun p => p.2) q).1) ((fun p => p.2) q).2
      -/
      exact (t't hq).2.2.2.2.1
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.refine_4
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        ⊢ Eq (μ (SDiff.sdiff s (Set.iUnion fun p => Set.iUnion fun h => p.2))) 0
      -/
    · convert μt' using 3
      /-
        case h.e'_2.h.e'_6.h.e'_4
        α : Type u_1
        ι : Type u_2
        inst✝⁴ : PseudoMetricSpace α
        inst✝³ : MeasurableSpace α
        inst✝² : OpensMeasurableSpace α
        inst✝¹ : SecondCountableTopology α
        μ : MeasureTheory.Measure α
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        C : NNReal
        h : ∀ (x : α), Filter.Frequently (fun r => LE.le (μ (Metric.closedBall x (HMul …
        s : Set α
        f : α → Set (Set α)
        fsubset : ∀ (x : α), Membership.mem s x → HasSubset.Subset (f x) ((fun x => se …
        ffine : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun t …
        t : Set (Prod Real (Prod α (Set α))) := setOf fun p => And (HasSubset.Subset p …
        A : ∀ (x : α), Membership.mem s x → ∀ (ε : Real), GT.gt ε 0 → Exists fun p =>  …
        t' : Set (Prod Real (Prod α (Set α)))
        t't : HasSubset.Subset t' t
        left✝ : t'.Countable
        t'_disj : t'.PairwiseDisjoint fun p => p.2.2
        μt' : Eq (μ (SDiff.sdiff s (Set.iUnion fun a => Set.iUnion fun h => a.2.2))) 0
        ⊢ Eq (Set.iUnion fun p => Set.iUnion fun h => p.2) (Set.iUnion fun a => Set.iU …
      -/
      rw [biUnion_image]
      /-
        🎉 no goals
      -/


