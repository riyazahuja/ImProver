/-- A family of additive subgroups on a ring `A` is a subgroups basis if it satisfies some
axioms ensuring there is a topology on `A` which is compatible with the ring structure and
admits this family as a basis of neighborhoods of zero. -/
structure RingSubgroupsBasis {A ι : Type*} [Ring A] (B : ι → AddSubgroup A) : Prop where
  /-- Condition for `B` to be a filter basis on `A`. -/
  inter : ∀ i j, ∃ k, B k ≤ B i ⊓ B j
  /-- For each set `B` in the submodule basis on `A`, there is another basis element `B'` such
   that the set-theoretic product `B' * B'` is in `B`. -/
  mul : ∀ i, ∃ j, (B j : Set A) * B j ⊆ B i
  /-- For any element `x : A` and any set `B` in the submodule basis on `A`,
    there is another basis element `B'` such that `B' * x` is in `B`. -/
  leftMul : ∀ x : A, ∀ i, ∃ j, (B j : Set A) ⊆ (x * ·) ⁻¹' B i
  /-- For any element `x : A` and any set `B` in the submodule basis on `A`,
    there is another basis element `B'` such that `x * B'` is in `B`. -/
  rightMul : ∀ x : A, ∀ i, ∃ j, (B j : Set A) ⊆ (· * x) ⁻¹' B i


theorem of_comm {A ι : Type*} [CommRing A] (B : ι → AddSubgroup A)
    (inter : ∀ i j, ∃ k, B k ≤ B i ⊓ B j) (mul : ∀ i, ∃ j, (B j : Set A) * B j ⊆ B i)
    (leftMul : ∀ x : A, ∀ i, ∃ j, (B j : Set A) ⊆ (fun y : A => x * y) ⁻¹' B i) :
    RingSubgroupsBasis B :=
  { inter
    mul
    leftMul
                                                          /-
                                                            A : Type u_3
                                                            ι : Type u_4
                                                            inst✝ : CommRing A
                                                            B : ι → AddSubgroup A
                                                            inter : ∀ (i j : ι), Exists fun k => LE.le (B k) (Min.min (B i) (B j))
                                                            mul : ∀ (i : ι), Exists fun j => HasSubset.Subset (HMul.hMul ↑(B j) ↑(B j)) ↑( …
                                                            leftMul : ∀ (x : A) (i : ι), Exists fun j => HasSubset.Subset (↑(B j)) (Set.pr …
                                                            x : A
                                                            i j : ι
                                                            hj : HasSubset.Subset (↑(B j)) (Set.preimage (fun y => HMul.hMul x y) ↑(B i))
                                                            ⊢ HasSubset.Subset (↑(B j)) (Set.preimage (fun x_1 => HMul.hMul x_1 x) ↑(B i))
                                                          -/
    rightMul := fun x i ↦ (leftMul x i).imp fun j hj ↦ by simpa only [mul_comm] using hj }
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Every subgroups basis on a ring leads to a ring filter basis. -/
def toRingFilterBasis [Nonempty ι] {B : ι → AddSubgroup A} (hB : RingSubgroupsBasis B) :
    RingFilterBasis A where
  sets := { U | ∃ i, U = B i }
  nonempty := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ (setOf fun U => Exists fun i => Eq U ↑(B i)).Nonempty
    -/
    inhabit ι
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      inhabited_h : Inhabited ι
      ⊢ (setOf fun U => Exists fun i => Eq U ↑(B i)).Nonempty
    -/
    exact ⟨B default, default, rfl⟩
    /-
      🎉 no goals
    -/
  inter_sets := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ {x y : Set A}, Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B i)) …
    -/
    rintro _ _ ⟨i, rfl⟩ ⟨j, rfl⟩
    /-
      case intro.intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i j : ι
      ⊢ Exists fun z => And (Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B …
    -/
    cases' hB.inter i j with k hk
    /-
      case intro.intro.intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i j k : ι
      hk : LE.le (B k) (Min.min (B i) (B j))
      ⊢ Exists fun z => And (Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B …
    -/
    use B k
    /-
      case h
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i j k : ι
      hk : LE.le (B k) (Min.min (B i) (B j))
      ⊢ And (Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B i)) ↑(B k)) (Ha …
    -/
    constructor
      /-
        case h.left
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i j k : ι
        hk : LE.le (B k) (Min.min (B i) (B j))
        ⊢ Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B i)) ↑(B k)
      -/
    · use k
      /-
        🎉 no goals
      -/
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i j k : ι
        hk : LE.le (B k) (Min.min (B i) (B j))
        ⊢ HasSubset.Subset (↑(B k)) (Inter.inter ↑(B i) ↑(B j))
      -/
    · exact hk
      /-
        🎉 no goals
      -/
  zero' := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ {U : Set A}, Membership.mem { sets := setOf fun U => Exists fun i => Eq U  …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i : ι
      ⊢ Membership.mem (↑(B i)) 0
    -/
    exact (B i).zero_mem
    /-
      🎉 no goals
    -/
  add' := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ {U : Set A}, Membership.mem { sets := setOf fun U => Exists fun i => Eq U  …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i : ι
      ⊢ Exists fun V => And (Membership.mem { sets := setOf fun U => Exists fun i => …
    -/
    use B i
    /-
      case h
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i : ι
      ⊢ And (Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), no …
    -/
    constructor
      /-
        case h.left
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        ⊢ Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), nonempt …
      -/
    · use i
      /-
        🎉 no goals
      -/
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        ⊢ HasSubset.Subset (HAdd.hAdd ↑(B i) ↑(B i)) ↑(B i)
      -/
    · rintro x ⟨y, y_in, z, z_in, rfl⟩
      /-
        case h.right.intro.intro.intro.intro
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        y : A
        y_in : Membership.mem (↑(B i)) y
        z : A
        z_in : Membership.mem (↑(B i)) z
        ⊢ Membership.mem (↑(B i)) ((fun x1 x2 => HAdd.hAdd x1 x2) y z)
      -/
      exact (B i).add_mem y_in z_in
      /-
        🎉 no goals
      -/
  neg' := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ {U : Set A}, Membership.mem { sets := setOf fun U => Exists fun i => Eq U  …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i : ι
      ⊢ Exists fun V => And (Membership.mem { sets := setOf fun U => Exists fun i => …
    -/
    use B i
    /-
      case h
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i : ι
      ⊢ And (Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), no …
    -/
    constructor
      /-
        case h.left
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        ⊢ Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), nonempt …
      -/
    · use i
      /-
        🎉 no goals
      -/
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        ⊢ HasSubset.Subset (↑(B i)) (Set.preimage (fun x => Neg.neg x) ↑(B i))
      -/
    · intro x x_in
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        x : A
        x_in : Membership.mem (↑(B i)) x
        ⊢ Membership.mem (Set.preimage (fun x => Neg.neg x) ↑(B i)) x
      -/
      exact (B i).neg_mem x_in
      /-
        🎉 no goals
      -/
  conj' := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ (x₀ : A) {U : Set A}, Membership.mem { sets := setOf fun U => Exists fun i …
    -/
    rintro x₀ _ ⟨i, rfl⟩
    /-
      case intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i : ι
      ⊢ Exists fun V => And (Membership.mem { sets := setOf fun U => Exists fun i => …
    -/
    use B i
    /-
      case h
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i : ι
      ⊢ And (Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), no …
    -/
    constructor
      /-
        case h.left
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        x₀ : A
        i : ι
        ⊢ Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), nonempt …
      -/
    · use i
      /-
        🎉 no goals
      -/
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        x₀ : A
        i : ι
        ⊢ HasSubset.Subset (↑(B i)) (Set.preimage (fun x => HAdd.hAdd (HAdd.hAdd x₀ x) …
      -/
    · simp
      /-
        🎉 no goals
      -/
  mul' := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ {U : Set A}, Membership.mem AddGroupFilterBasis.toFilterBasis.sets U → Exi …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i : ι
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    cases' hB.mul i with k hk
    /-
      case intro.intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i k : ι
      hk : HasSubset.Subset (HMul.hMul ↑(B k) ↑(B k)) ↑(B i)
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    use B k
    /-
      case h
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      i k : ι
      hk : HasSubset.Subset (HMul.hMul ↑(B k) ↑(B k)) ↑(B i)
      ⊢ And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B k)) (HasSubse …
    -/
    constructor
      /-
        case h.left
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i k : ι
        hk : HasSubset.Subset (HMul.hMul ↑(B k) ↑(B k)) ↑(B i)
        ⊢ Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B k)
      -/
    · use k
      /-
        🎉 no goals
      -/
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i k : ι
        hk : HasSubset.Subset (HMul.hMul ↑(B k) ↑(B k)) ↑(B i)
        ⊢ HasSubset.Subset (HMul.hMul ↑(B k) ↑(B k)) ↑(B i)
      -/
    · exact hk
      /-
        🎉 no goals
      -/
  mul_left' := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ (x₀ : A) {U : Set A}, Membership.mem AddGroupFilterBasis.toFilterBasis.set …
    -/
    rintro x₀ _ ⟨i, rfl⟩
    /-
      case intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i : ι
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    cases' hB.leftMul x₀ i with k hk
    /-
      case intro.intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i k : ι
      hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x₀ x) ↑(B i))
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    use B k
    /-
      case h
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i k : ι
      hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x₀ x) ↑(B i))
      ⊢ And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B k)) (HasSubse …
    -/
    constructor
      /-
        case h.left
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        x₀ : A
        i k : ι
        hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x₀ x) ↑(B i))
        ⊢ Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B k)
      -/
    · use k
      /-
        🎉 no goals
      -/
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        x₀ : A
        i k : ι
        hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x₀ x) ↑(B i))
        ⊢ HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x₀ x) ↑(B i))
      -/
    · exact hk
      /-
        🎉 no goals
      -/
  mul_right' := by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ (x₀ : A) {U : Set A}, Membership.mem AddGroupFilterBasis.toFilterBasis.set …
    -/
    rintro x₀ _ ⟨i, rfl⟩
    /-
      case intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i : ι
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    cases' hB.rightMul x₀ i with k hk
    /-
      case intro.intro
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i k : ι
      hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x x₀) ↑(B i))
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    use B k
    /-
      case h
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      x₀ : A
      i k : ι
      hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x x₀) ↑(B i))
      ⊢ And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B k)) (HasSubse …
    -/
    constructor
      /-
        case h.left
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        x₀ : A
        i k : ι
        hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x x₀) ↑(B i))
        ⊢ Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B k)
      -/
    · use k
      /-
        🎉 no goals
      -/
      /-
        case h.right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        x₀ : A
        i k : ι
        hk : HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x x₀) ↑(B i))
        ⊢ HasSubset.Subset (↑(B k)) (Set.preimage (fun x => HMul.hMul x x₀) ↑(B i))
      -/
    · exact hk
      /-
        🎉 no goals
      -/


theorem mem_addGroupFilterBasis_iff {V : Set A} :
    V ∈ hB.toRingFilterBasis.toAddGroupFilterBasis ↔ ∃ i, V = B i :=
  Iff.rfl


theorem mem_addGroupFilterBasis (i) : (B i : Set A) ∈ hB.toRingFilterBasis.toAddGroupFilterBasis :=
  ⟨i, rfl⟩


/-- The topology defined from a subgroups basis, admitting the given subgroups as a basis
of neighborhoods of zero. -/
def topology : TopologicalSpace A :=
  hB.toRingFilterBasis.toAddGroupFilterBasis.topology


theorem hasBasis_nhds_zero : HasBasis (@nhds A hB.topology 0) (fun _ => True) fun i => B i :=
  ⟨by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      ⊢ ∀ (t : Set A), Iff (Membership.mem (nhds 0) t) (Exists fun i => And True (Ha …
    -/
    intro s
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      s : Set A
      ⊢ Iff (Membership.mem (nhds 0) s) (Exists fun i => And True (HasSubset.Subset  …
    -/
    rw [hB.toRingFilterBasis.toAddGroupFilterBasis.nhds_zero_hasBasis.mem_iff]
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      s : Set A
      ⊢ Iff (Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBas …
    -/
    constructor
      /-
        case mp
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        s : Set A
        ⊢ (Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBasis i …
      -/
    · rintro ⟨-, ⟨i, rfl⟩, hi⟩
      /-
        case mp.intro.intro.intro
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        s : Set A
        i : ι
        hi : HasSubset.Subset (id ↑(B i)) s
        ⊢ Exists fun i => And True (HasSubset.Subset (↑(B i)) s)
      -/
      exact ⟨i, trivial, hi⟩
      /-
        🎉 no goals
      -/
      /-
        case mpr
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        s : Set A
        ⊢ (Exists fun i => And True (HasSubset.Subset (↑(B i)) s)) → Exists fun i => A …
      -/
    · rintro ⟨i, -, hi⟩
      /-
        case mpr.intro.intro
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        s : Set A
        i : ι
        hi : HasSubset.Subset (↑(B i)) s
        ⊢ Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBasis i) …
      -/
      exact ⟨B i, ⟨i, rfl⟩, hi⟩⟩
      /-
        🎉 no goals
      -/


theorem hasBasis_nhds (a : A) :
    HasBasis (@nhds A hB.topology a) (fun _ => True) fun i => { b | b - a ∈ B i } :=
  ⟨by
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      a : A
      ⊢ ∀ (t : Set A), Iff (Membership.mem (nhds a) t) (Exists fun i => And True (Ha …
    -/
    intro s
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      a : A
      s : Set A
      ⊢ Iff (Membership.mem (nhds a) s) (Exists fun i => And True (HasSubset.Subset  …
    -/
    rw [(hB.toRingFilterBasis.toAddGroupFilterBasis.nhds_hasBasis a).mem_iff]
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      a : A
      s : Set A
      ⊢ Iff (Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBas …
    -/
    simp only [true_and]
    /-
      A : Type u_1
      ι : Type u_2
      inst✝¹ : Ring A
      inst✝ : Nonempty ι
      B : ι → AddSubgroup A
      hB : RingSubgroupsBasis B
      a : A
      s : Set A
      ⊢ Iff (Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBas …
    -/
    constructor
      /-
        case mp
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        ⊢ (Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBasis i …
      -/
    · rintro ⟨-, ⟨i, rfl⟩, hi⟩
      /-
        case mp.intro.intro.intro
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        i : ι
        hi : HasSubset.Subset (Set.image (fun y => HAdd.hAdd a y) ↑(B i)) s
        ⊢ Exists fun i => HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub. …
      -/
      use i
      suffices h : { b : A | b - a ∈ B i } = (fun y => a + y) '' ↑(B i) by
        rw [h]
        assumption
      /-
        case h
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        i : ι
        hi : HasSubset.Subset (Set.image (fun y => HAdd.hAdd a y) ↑(B i)) s
        ⊢ Eq (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) (Set.image (fun y = …
      -/
      simp only [image_add_left, neg_add_eq_sub]
      /-
        case h
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        i : ι
        hi : HasSubset.Subset (Set.image (fun y => HAdd.hAdd a y) ↑(B i)) s
        ⊢ Eq (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) (Set.preimage (fun  …
      -/
      ext b
      /-
        case h.h
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        i : ι
        hi : HasSubset.Subset (Set.image (fun y => HAdd.hAdd a y) ↑(B i)) s
        b : A
        ⊢ Iff (Membership.mem (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) b) …
      -/
      simp
      /-
        🎉 no goals
      -/
      /-
        case mpr
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        ⊢ (Exists fun i => HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub …
      -/
    · rintro ⟨i, hi⟩
      /-
        case mpr.intro
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        i : ι
        hi : HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) s
        ⊢ Exists fun i => And (Membership.mem RingFilterBasis.toAddGroupFilterBasis i) …
      -/
      use B i
      /-
        case h
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        a : A
        s : Set A
        i : ι
        hi : HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) s
        ⊢ And (Membership.mem RingFilterBasis.toAddGroupFilterBasis ↑(B i)) (HasSubset …
      -/
      constructor
        /-
          case h.left
          A : Type u_1
          ι : Type u_2
          inst✝¹ : Ring A
          inst✝ : Nonempty ι
          B : ι → AddSubgroup A
          hB : RingSubgroupsBasis B
          a : A
          s : Set A
          i : ι
          hi : HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) s
          ⊢ Membership.mem RingFilterBasis.toAddGroupFilterBasis ↑(B i)
        -/
      · use i
        /-
          🎉 no goals
        -/
        /-
          case h.right
          A : Type u_1
          ι : Type u_2
          inst✝¹ : Ring A
          inst✝ : Nonempty ι
          B : ι → AddSubgroup A
          hB : RingSubgroupsBasis B
          a : A
          s : Set A
          i : ι
          hi : HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) s
          ⊢ HasSubset.Subset (Set.image (fun y => HAdd.hAdd a y) ↑(B i)) s
        -/
      · rw [image_subset_iff]
        /-
          case h.right
          A : Type u_1
          ι : Type u_2
          inst✝¹ : Ring A
          inst✝ : Nonempty ι
          B : ι → AddSubgroup A
          hB : RingSubgroupsBasis B
          a : A
          s : Set A
          i : ι
          hi : HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) s
          ⊢ HasSubset.Subset (↑(B i)) (Set.preimage (fun y => HAdd.hAdd a y) s)
        -/
        rintro b b_in
        /-
          case h.right
          A : Type u_1
          ι : Type u_2
          inst✝¹ : Ring A
          inst✝ : Nonempty ι
          B : ι → AddSubgroup A
          hB : RingSubgroupsBasis B
          a : A
          s : Set A
          i : ι
          hi : HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) s
          b : A
          b_in : Membership.mem (↑(B i)) b
          ⊢ Membership.mem (Set.preimage (fun y => HAdd.hAdd a y) s) b
        -/
        apply hi
        /-
          case h.right.a
          A : Type u_1
          ι : Type u_2
          inst✝¹ : Ring A
          inst✝ : Nonempty ι
          B : ι → AddSubgroup A
          hB : RingSubgroupsBasis B
          a : A
          s : Set A
          i : ι
          hi : HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) s
          b : A
          b_in : Membership.mem (↑(B i)) b
          ⊢ Membership.mem (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) ((fun y …
        -/
        simpa using b_in⟩
        /-
          🎉 no goals
        -/


/-- Given a subgroups basis, the basis elements as open additive subgroups in the associated
topology. -/
def openAddSubgroup (i : ι) : @OpenAddSubgroup A _ hB.topology :=
  -- Porting note: failed to synthesize instance `TopologicalSpace A`
  let _ := hB.topology
  { B i with
    isOpen' := by
      /-
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        x✝ : TopologicalSpace A := hB.topology
        ⊢ IsOpen __src✝.carrier
      -/
      rw [isOpen_iff_mem_nhds]
      /-
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        x✝ : TopologicalSpace A := hB.topology
        ⊢ ∀ (x : A), Membership.mem __src✝.carrier x → Membership.mem (nhds x) __src✝. …
      -/
      intro a a_in
      /-
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        x✝ : TopologicalSpace A := hB.topology
        a : A
        a_in : Membership.mem __src✝.carrier a
        ⊢ Membership.mem (nhds a) __src✝.carrier
      -/
      rw [(hB.hasBasis_nhds a).mem_iff]
      /-
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        x✝ : TopologicalSpace A := hB.topology
        a : A
        a_in : Membership.mem __src✝.carrier a
        ⊢ Exists fun i => And True (HasSubset.Subset (setOf fun b => Membership.mem (B …
      -/
      use i, trivial
      /-
        case right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        x✝ : TopologicalSpace A := hB.topology
        a : A
        a_in : Membership.mem __src✝.carrier a
        ⊢ HasSubset.Subset (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) __src …
      -/
      rintro b b_in
      /-
        case right
        A : Type u_1
        ι : Type u_2
        inst✝¹ : Ring A
        inst✝ : Nonempty ι
        B : ι → AddSubgroup A
        hB : RingSubgroupsBasis B
        i : ι
        x✝ : TopologicalSpace A := hB.topology
        a : A
        a_in : Membership.mem __src✝.carrier a
        b : A
        b_in : Membership.mem (setOf fun b => Membership.mem (B i) (HSub.hSub b a)) b
        ⊢ Membership.mem __src✝.carrier b
      -/
      simpa using (B i).add_mem a_in b_in }
      /-
        🎉 no goals
      -/

-- see Note [nonarchimedean non instances]

theorem nonarchimedean : @NonarchimedeanRing A _ hB.topology := by
  /-
    A : Type u_1
    ι : Type u_2
    inst✝¹ : Ring A
    inst✝ : Nonempty ι
    B : ι → AddSubgroup A
    hB : RingSubgroupsBasis B
    ⊢ NonarchimedeanRing A
  -/
  letI := hB.topology
  /-
    A : Type u_1
    ι : Type u_2
    inst✝¹ : Ring A
    inst✝ : Nonempty ι
    B : ι → AddSubgroup A
    hB : RingSubgroupsBasis B
    this : TopologicalSpace A := hB.topology
    ⊢ NonarchimedeanRing A
  -/
  constructor
  /-
    case is_nonarchimedean
    A : Type u_1
    ι : Type u_2
    inst✝¹ : Ring A
    inst✝ : Nonempty ι
    B : ι → AddSubgroup A
    hB : RingSubgroupsBasis B
    this : TopologicalSpace A := hB.topology
    ⊢ ∀ (U : Set A), Membership.mem (nhds 0) U → Exists fun V => HasSubset.Subset  …
  -/
  intro U hU
  /-
    case is_nonarchimedean
    A : Type u_1
    ι : Type u_2
    inst✝¹ : Ring A
    inst✝ : Nonempty ι
    B : ι → AddSubgroup A
    hB : RingSubgroupsBasis B
    this : TopologicalSpace A := hB.topology
    U : Set A
    hU : Membership.mem (nhds 0) U
    ⊢ Exists fun V => HasSubset.Subset (↑V) U
  -/
  obtain ⟨i, -, hi : (B i : Set A) ⊆ U⟩ := hB.hasBasis_nhds_zero.mem_iff.mp hU
  /-
    case is_nonarchimedean.intro.intro
    A : Type u_1
    ι : Type u_2
    inst✝¹ : Ring A
    inst✝ : Nonempty ι
    B : ι → AddSubgroup A
    hB : RingSubgroupsBasis B
    this : TopologicalSpace A := hB.topology
    U : Set A
    hU : Membership.mem (nhds 0) U
    i : ι
    hi : HasSubset.Subset (↑(B i)) U
    ⊢ Exists fun V => HasSubset.Subset (↑V) U
  -/
  exact ⟨hB.openAddSubgroup i, hi⟩
  /-
    🎉 no goals
  -/


/-- A family of submodules in a commutative `R`-algebra `A` is a submodules basis if it satisfies
some axioms ensuring there is a topology on `A` which is compatible with the ring structure and
admits this family as a basis of neighborhoods of zero. -/
structure SubmodulesRingBasis (B : ι → Submodule R A) : Prop where
  /-- Condition for `B` to be a filter basis on `A`. -/
  inter : ∀ i j, ∃ k, B k ≤ B i ⊓ B j
  /-- For any element `a : A` and any set `B` in the submodule basis on `A`,
    there is another basis element `B'` such that `a • B'` is in `B`. -/
  leftMul : ∀ (a : A) (i), ∃ j, a • B j ≤ B i
  /-- For each set `B` in the submodule basis on `A`, there is another basis element `B'` such
    that the set-theoretic product `B' * B'` is in `B`. -/
  mul : ∀ i, ∃ j, (B j : Set A) * B j ⊆ B i


theorem toRing_subgroups_basis (hB : SubmodulesRingBasis B) :
    RingSubgroupsBasis fun i => (B i).toAddSubgroup := by
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    B : ι → Submodule R A
    hB : SubmodulesRingBasis B
    ⊢ RingSubgroupsBasis fun i => (B i).toAddSubgroup
  -/
  apply RingSubgroupsBasis.of_comm (fun i => (B i).toAddSubgroup) hB.inter hB.mul
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    B : ι → Submodule R A
    hB : SubmodulesRingBasis B
    ⊢ ∀ (x : A) (i : ι), Exists fun j => HasSubset.Subset (↑(B j).toAddSubgroup) ( …
  -/
  intro a i
  /-
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    B : ι → Submodule R A
    hB : SubmodulesRingBasis B
    a : A
    i : ι
    ⊢ Exists fun j => HasSubset.Subset (↑(B j).toAddSubgroup) (Set.preimage (fun y …
  -/
  rcases hB.leftMul a i with ⟨j, hj⟩
  /-
    case intro
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    B : ι → Submodule R A
    hB : SubmodulesRingBasis B
    a : A
    i j : ι
    hj : LE.le (HSMul.hSMul a (B j)) (B i)
    ⊢ Exists fun j => HasSubset.Subset (↑(B j).toAddSubgroup) (Set.preimage (fun y …
  -/
  use j
  /-
    case h
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    B : ι → Submodule R A
    hB : SubmodulesRingBasis B
    a : A
    i j : ι
    hj : LE.le (HSMul.hSMul a (B j)) (B i)
    ⊢ HasSubset.Subset (↑(B j).toAddSubgroup) (Set.preimage (fun y => HMul.hMul a  …
  -/
  rintro b (b_in : b ∈ B j)
  /-
    case h
    ι : Type u_1
    R : Type u_2
    A : Type u_3
    inst✝² : CommRing R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    B : ι → Submodule R A
    hB : SubmodulesRingBasis B
    a : A
    i j : ι
    hj : LE.le (HSMul.hSMul a (B j)) (B i)
    b : A
    b_in : Membership.mem (B j) b
    ⊢ Membership.mem (Set.preimage (fun y => HMul.hMul a y) ↑(B i).toAddSubgroup) b
  -/
  exact hj ⟨b, b_in, rfl⟩
  /-
    🎉 no goals
  -/


/-- The topology associated to a basis of submodules in an algebra. -/
def topology [Nonempty ι] (hB : SubmodulesRingBasis B) : TopologicalSpace A :=
  hB.toRing_subgroups_basis.topology


/-- A family of submodules in an `R`-module `M` is a submodules basis if it satisfies
some axioms ensuring there is a topology on `M` which is compatible with the module structure and
admits this family as a basis of neighborhoods of zero. -/
structure SubmodulesBasis [TopologicalSpace R] (B : ι → Submodule R M) : Prop where
  /-- Condition for `B` to be a filter basis on `M`. -/
  inter : ∀ i j, ∃ k, B k ≤ B i ⊓ B j
  /-- For any element `m : M` and any set `B` in the basis, `a • m` lies in `B` for all
    `a` sufficiently close to `0`. -/
  smul : ∀ (m : M) (i : ι), ∀ᶠ a in 𝓝 (0 : R), a • m ∈ B i


/-- The image of a submodules basis is a module filter basis. -/
def toModuleFilterBasis : ModuleFilterBasis R M where
  sets := { U | ∃ i, U = B i }
  nonempty := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ (setOf fun U => Exists fun i => Eq U ↑(B i)).Nonempty
    -/
    inhabit ι
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      inhabited_h : Inhabited ι
      ⊢ (setOf fun U => Exists fun i => Eq U ↑(B i)).Nonempty
    -/
    exact ⟨B default, default, rfl⟩
    /-
      🎉 no goals
    -/
  inter_sets := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ {x y : Set M}, Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B i)) …
    -/
    rintro _ _ ⟨i, rfl⟩ ⟨j, rfl⟩
    /-
      case intro.intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i j : ι
      ⊢ Exists fun z => And (Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B …
    -/
    cases' hB.inter i j with k hk
    /-
      case intro.intro.intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i j k : ι
      hk : LE.le (B k) (Min.min (B i) (B j))
      ⊢ Exists fun z => And (Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B …
    -/
    use B k
    /-
      case h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i j k : ι
      hk : LE.le (B k) (Min.min (B i) (B j))
      ⊢ And (Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B i)) ↑(B k)) (Ha …
    -/
    constructor
      /-
        case h.left
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i j k : ι
        hk : LE.le (B k) (Min.min (B i) (B j))
        ⊢ Membership.mem (setOf fun U => Exists fun i => Eq U ↑(B i)) ↑(B k)
      -/
    · use k
      /-
        🎉 no goals
      -/
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i j k : ι
        hk : LE.le (B k) (Min.min (B i) (B j))
        ⊢ HasSubset.Subset (↑(B k)) (Inter.inter ↑(B i) ↑(B j))
      -/
    · exact hk
      /-
        🎉 no goals
      -/
  zero' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ {U : Set M}, Membership.mem { sets := setOf fun U => Exists fun i => Eq U  …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i : ι
      ⊢ Membership.mem (↑(B i)) 0
    -/
    exact (B i).zero_mem
    /-
      🎉 no goals
    -/
  add' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ {U : Set M}, Membership.mem { sets := setOf fun U => Exists fun i => Eq U  …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i : ι
      ⊢ Exists fun V => And (Membership.mem { sets := setOf fun U => Exists fun i => …
    -/
    use B i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i : ι
      ⊢ And (Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), no …
    -/
    constructor
      /-
        case h.left
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        ⊢ Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), nonempt …
      -/
    · use i
      /-
        🎉 no goals
      -/
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        ⊢ HasSubset.Subset (HAdd.hAdd ↑(B i) ↑(B i)) ↑(B i)
      -/
    · rintro x ⟨y, y_in, z, z_in, rfl⟩
      /-
        case h.right.intro.intro.intro.intro
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        y : M
        y_in : Membership.mem (↑(B i)) y
        z : M
        z_in : Membership.mem (↑(B i)) z
        ⊢ Membership.mem (↑(B i)) ((fun x1 x2 => HAdd.hAdd x1 x2) y z)
      -/
      exact (B i).add_mem y_in z_in
      /-
        🎉 no goals
      -/
  neg' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ {U : Set M}, Membership.mem { sets := setOf fun U => Exists fun i => Eq U  …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i : ι
      ⊢ Exists fun V => And (Membership.mem { sets := setOf fun U => Exists fun i => …
    -/
    use B i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i : ι
      ⊢ And (Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), no …
    -/
    constructor
      /-
        case h.left
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        ⊢ Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), nonempt …
      -/
    · use i
      /-
        🎉 no goals
      -/
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        ⊢ HasSubset.Subset (↑(B i)) (Set.preimage (fun x => Neg.neg x) ↑(B i))
      -/
    · intro x x_in
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        x : M
        x_in : Membership.mem (↑(B i)) x
        ⊢ Membership.mem (Set.preimage (fun x => Neg.neg x) ↑(B i)) x
      -/
      exact (B i).neg_mem x_in
      /-
        🎉 no goals
      -/
  conj' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ (x₀ : M) {U : Set M}, Membership.mem { sets := setOf fun U => Exists fun i …
    -/
    rintro x₀ _ ⟨i, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      x₀ : M
      i : ι
      ⊢ Exists fun V => And (Membership.mem { sets := setOf fun U => Exists fun i => …
    -/
    use B i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      x₀ : M
      i : ι
      ⊢ And (Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), no …
    -/
    constructor
      /-
        case h.left
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        x₀ : M
        i : ι
        ⊢ Membership.mem { sets := setOf fun U => Exists fun i => Eq U ↑(B i), nonempt …
      -/
    · use i
      /-
        🎉 no goals
      -/
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        x₀ : M
        i : ι
        ⊢ HasSubset.Subset (↑(B i)) (Set.preimage (fun x => HAdd.hAdd (HAdd.hAdd x₀ x) …
      -/
    · simp
      /-
        🎉 no goals
      -/
  smul' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.sets U → Exi …
    -/
    rintro _ ⟨i, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i : ι
      ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (Exists fun W => And (Member …
    -/
    use univ
    /-
      case h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      i : ι
      ⊢ And (Membership.mem (nhds 0) Set.univ) (Exists fun W => And (Membership.mem  …
    -/
    constructor
      /-
        case h.left
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        ⊢ Membership.mem (nhds 0) Set.univ
      -/
    · exact univ_mem
      /-
        🎉 no goals
      -/
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        ⊢ Exists fun W => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets W …
      -/
    · use B i
      /-
        case h
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        ⊢ And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B i)) (HasSubse …
      -/
      constructor
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing A
          inst✝⁴ : Algebra R A
          M : Type u_4
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : TopologicalSpace R
          inst✝ : Nonempty ι
          B : ι → Submodule R M
          hB : SubmodulesBasis B
          i : ι
          ⊢ Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B i)
        -/
      · use i
        /-
          🎉 no goals
        -/
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing A
          inst✝⁴ : Algebra R A
          M : Type u_4
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : TopologicalSpace R
          inst✝ : Nonempty ι
          B : ι → Submodule R M
          hB : SubmodulesBasis B
          i : ι
          ⊢ HasSubset.Subset (HSMul.hSMul Set.univ ↑(B i)) ↑(B i)
        -/
      · rintro _ ⟨a, -, m, hm, rfl⟩
        /-
          case h.right.intro.intro.intro.intro
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing A
          inst✝⁴ : Algebra R A
          M : Type u_4
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : TopologicalSpace R
          inst✝ : Nonempty ι
          B : ι → Submodule R M
          hB : SubmodulesBasis B
          i : ι
          a : R
          m : M
          hm : Membership.mem (↑(B i)) m
          ⊢ Membership.mem (↑(B i)) ((fun x1 x2 => HSMul.hSMul x1 x2) a m)
        -/
        exact (B i).smul_mem _ hm
        /-
          🎉 no goals
        -/
  smul_left' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ (x₀ : R) {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.set …
    -/
    rintro x₀ _ ⟨i, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      x₀ : R
      i : ι
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    use B i
    /-
      case h
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      x₀ : R
      i : ι
      ⊢ And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B i)) (HasSubse …
    -/
    constructor
      /-
        case h.left
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        x₀ : R
        i : ι
        ⊢ Membership.mem AddGroupFilterBasis.toFilterBasis.sets ↑(B i)
      -/
    · use i
      /-
        🎉 no goals
      -/
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        x₀ : R
        i : ι
        ⊢ HasSubset.Subset (↑(B i)) (Set.preimage (fun x => HSMul.hSMul x₀ x) ↑(B i))
      -/
    · intro m
      /-
        case h.right
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        x₀ : R
        i : ι
        m : M
        ⊢ Membership.mem (↑(B i)) m → Membership.mem (Set.preimage (fun x => HSMul.hSM …
      -/
      exact (B i).smul_mem _
      /-
        🎉 no goals
      -/
  smul_right' := by
    /-
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      ⊢ ∀ (m₀ : M) {U : Set M}, Membership.mem AddGroupFilterBasis.toFilterBasis.set …
    -/
    rintro m₀ _ ⟨i, rfl⟩
    /-
      case intro
      ι : Type u_1
      R : Type u_2
      A : Type u_3
      inst✝⁶ : CommRing R
      inst✝⁵ : CommRing A
      inst✝⁴ : Algebra R A
      M : Type u_4
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : TopologicalSpace R
      inst✝ : Nonempty ι
      B : ι → Submodule R M
      hB : SubmodulesBasis B
      m₀ : M
      i : ι
      ⊢ Filter.Eventually (fun x => Membership.mem (↑(B i)) (HSMul.hSMul x m₀)) (nhd …
    -/
    exact hB.smul m₀ i
    /-
      🎉 no goals
    -/


/-- The topology associated to a basis of submodules in a module. -/
def topology : TopologicalSpace M :=
  hB.toModuleFilterBasis.toAddGroupFilterBasis.topology


/-- Given a submodules basis, the basis elements as open additive subgroups in the associated
topology. -/
def openAddSubgroup (i : ι) : @OpenAddSubgroup M _ hB.topology :=
  let _ := hB.topology -- Porting note: failed to synthesize instance `TopologicalSpace A`
  { (B i).toAddSubgroup with
    isOpen' := by
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        x✝ : TopologicalSpace M := hB.topology
        ⊢ IsOpen __src✝.carrier
      -/
      letI := hB.topology
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        x✝ : TopologicalSpace M := hB.topology
        this : TopologicalSpace M := hB.topology
        ⊢ IsOpen __src✝.carrier
      -/
      rw [isOpen_iff_mem_nhds]
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        x✝ : TopologicalSpace M := hB.topology
        this : TopologicalSpace M := hB.topology
        ⊢ ∀ (x : M), Membership.mem __src✝.carrier x → Membership.mem (nhds x) __src✝. …
      -/
      intro a a_in
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        x✝ : TopologicalSpace M := hB.topology
        this : TopologicalSpace M := hB.topology
        a : M
        a_in : Membership.mem __src✝.carrier a
        ⊢ Membership.mem (nhds a) __src✝.carrier
      -/
      rw [(hB.toModuleFilterBasis.toAddGroupFilterBasis.nhds_hasBasis a).mem_iff]
      /-
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        x✝ : TopologicalSpace M := hB.topology
        this : TopologicalSpace M := hB.topology
        a : M
        a_in : Membership.mem __src✝.carrier a
        ⊢ Exists fun i => And (Membership.mem hB.toModuleFilterBasis.toAddGroupFilterB …
      -/
      use B i
      /-
        case h
        ι : Type u_1
        R : Type u_2
        A : Type u_3
        inst✝⁶ : CommRing R
        inst✝⁵ : CommRing A
        inst✝⁴ : Algebra R A
        M : Type u_4
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : TopologicalSpace R
        inst✝ : Nonempty ι
        B : ι → Submodule R M
        hB : SubmodulesBasis B
        i : ι
        x✝ : TopologicalSpace M := hB.topology
        this : TopologicalSpace M := hB.topology
        a : M
        a_in : Membership.mem __src✝.carrier a
        ⊢ And (Membership.mem hB.toModuleFilterBasis.toAddGroupFilterBasis ↑(B i)) (Ha …
      -/
      constructor
        /-
          case h.left
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing A
          inst✝⁴ : Algebra R A
          M : Type u_4
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : TopologicalSpace R
          inst✝ : Nonempty ι
          B : ι → Submodule R M
          hB : SubmodulesBasis B
          i : ι
          x✝ : TopologicalSpace M := hB.topology
          this : TopologicalSpace M := hB.topology
          a : M
          a_in : Membership.mem __src✝.carrier a
          ⊢ Membership.mem hB.toModuleFilterBasis.toAddGroupFilterBasis ↑(B i)
        -/
      · use i
        /-
          🎉 no goals
        -/
        /-
          case h.right
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing A
          inst✝⁴ : Algebra R A
          M : Type u_4
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : TopologicalSpace R
          inst✝ : Nonempty ι
          B : ι → Submodule R M
          hB : SubmodulesBasis B
          i : ι
          x✝ : TopologicalSpace M := hB.topology
          this : TopologicalSpace M := hB.topology
          a : M
          a_in : Membership.mem __src✝.carrier a
          ⊢ HasSubset.Subset (Set.image (fun y => HAdd.hAdd a y) ↑(B i)) __src✝.carrier
        -/
      · rintro - ⟨b, b_in, rfl⟩
        /-
          case h.right.intro.intro
          ι : Type u_1
          R : Type u_2
          A : Type u_3
          inst✝⁶ : CommRing R
          inst✝⁵ : CommRing A
          inst✝⁴ : Algebra R A
          M : Type u_4
          inst✝³ : AddCommGroup M
          inst✝² : Module R M
          inst✝¹ : TopologicalSpace R
          inst✝ : Nonempty ι
          B : ι → Submodule R M
          hB : SubmodulesBasis B
          i : ι
          x✝ : TopologicalSpace M := hB.topology
          this : TopologicalSpace M := hB.topology
          a : M
          a_in : Membership.mem __src✝.carrier a
          b : M
          b_in : Membership.mem (↑(B i)) b
          ⊢ Membership.mem __src✝.carrier ((fun y => HAdd.hAdd a y) b)
        -/
        exact (B i).add_mem a_in b_in }
        /-
          🎉 no goals
        -/

-- see Note [nonarchimedean non instances]

theorem nonarchimedean (hB : SubmodulesBasis B) : @NonarchimedeanAddGroup M _ hB.topology := by
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : TopologicalSpace R
    inst✝ : Nonempty ι
    B : ι → Submodule R M
    hB : SubmodulesBasis B
    ⊢ NonarchimedeanAddGroup M
  -/
  letI := hB.topology
  /-
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : TopologicalSpace R
    inst✝ : Nonempty ι
    B : ι → Submodule R M
    hB : SubmodulesBasis B
    this : TopologicalSpace M := hB.topology
    ⊢ NonarchimedeanAddGroup M
  -/
  constructor
  /-
    case is_nonarchimedean
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : TopologicalSpace R
    inst✝ : Nonempty ι
    B : ι → Submodule R M
    hB : SubmodulesBasis B
    this : TopologicalSpace M := hB.topology
    ⊢ ∀ (U : Set M), Membership.mem (nhds 0) U → Exists fun V => HasSubset.Subset  …
  -/
  intro U hU
  obtain ⟨-, ⟨i, rfl⟩, hi : (B i : Set M) ⊆ U⟩ :=
    hB.toModuleFilterBasis.toAddGroupFilterBasis.nhds_zero_hasBasis.mem_iff.mp hU
  /-
    case is_nonarchimedean.intro.intro.intro
    ι : Type u_1
    R : Type u_2
    inst✝⁴ : CommRing R
    M : Type u_4
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : TopologicalSpace R
    inst✝ : Nonempty ι
    B : ι → Submodule R M
    hB : SubmodulesBasis B
    this : TopologicalSpace M := hB.topology
    U : Set M
    hU : Membership.mem (nhds 0) U
    i : ι
    hi : HasSubset.Subset (↑(B i)) U
    ⊢ Exists fun V => HasSubset.Subset (↑V) U
  -/
  exact ⟨hB.openAddSubgroup i, hi⟩
  /-
    🎉 no goals
  -/


theorem SubmodulesRingBasis.toSubmodulesBasis : SubmodulesBasis B :=
  { inter := hB.inter
    smul := hsmul }


/-- Given a ring filter basis on a commutative ring `R`, define a compatibility condition
on a family of submodules of an `R`-module `M`. This compatibility condition allows to get
a topological module structure. -/
structure RingFilterBasis.SubmodulesBasis (BR : RingFilterBasis R) (B : ι → Submodule R M) :
    Prop where
  /-- Condition for `B` to be a filter basis on `M`. -/
  inter : ∀ i j, ∃ k, B k ≤ B i ⊓ B j
  /-- For any element `m : M` and any set `B i` in the submodule basis on `M`,
    there is a `U` in the ring filter basis on `R` such that `U * m` is in `B i`. -/
  smul : ∀ (m : M) (i : ι), ∃ U ∈ BR, U ⊆ (· • m) ⁻¹' B i


theorem RingFilterBasis.submodulesBasisIsBasis (BR : RingFilterBasis R) {B : ι → Submodule R M}
    (hB : BR.SubmodulesBasis B) : @_root_.SubmodulesBasis ι R _ M _ _ BR.topology B :=
  let _ := BR.topology -- Porting note: failed to synthesize instance `TopologicalSpace R`
  { inter := hB.inter
    smul := by
      /-
        ι : Type u_1
        R : Type u_2
        inst✝² : CommRing R
        M : Type u_4
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        B : ι → Submodule R M
        hB : BR.SubmodulesBasis B
        x✝ : TopologicalSpace R := BR.topology
        ⊢ ∀ (m : M) (i : ι), Filter.Eventually (fun a => Membership.mem (B i) (HSMul.h …
      -/
      letI := BR.topology
      /-
        ι : Type u_1
        R : Type u_2
        inst✝² : CommRing R
        M : Type u_4
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        B : ι → Submodule R M
        hB : BR.SubmodulesBasis B
        x✝ : TopologicalSpace R := BR.topology
        this : TopologicalSpace R := BR.topology
        ⊢ ∀ (m : M) (i : ι), Filter.Eventually (fun a => Membership.mem (B i) (HSMul.h …
      -/
      intro m i
      /-
        ι : Type u_1
        R : Type u_2
        inst✝² : CommRing R
        M : Type u_4
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        B : ι → Submodule R M
        hB : BR.SubmodulesBasis B
        x✝ : TopologicalSpace R := BR.topology
        this : TopologicalSpace R := BR.topology
        m : M
        i : ι
        ⊢ Filter.Eventually (fun a => Membership.mem (B i) (HSMul.hSMul a m)) (nhds 0)
      -/
      rcases hB.smul m i with ⟨V, V_in, hV⟩
      /-
        case intro.intro
        ι : Type u_1
        R : Type u_2
        inst✝² : CommRing R
        M : Type u_4
        inst✝¹ : AddCommGroup M
        inst✝ : Module R M
        BR : RingFilterBasis R
        B : ι → Submodule R M
        hB : BR.SubmodulesBasis B
        x✝ : TopologicalSpace R := BR.topology
        this : TopologicalSpace R := BR.topology
        m : M
        i : ι
        V : Set R
        V_in : Membership.mem BR V
        hV : HasSubset.Subset V (Set.preimage (fun x => HSMul.hSMul x m) ↑(B i))
        ⊢ Filter.Eventually (fun a => Membership.mem (B i) (HSMul.hSMul a m)) (nhds 0)
      -/
      exact mem_of_superset (BR.toAddGroupFilterBasis.mem_nhds_zero V_in) hV }
      /-
        🎉 no goals
      -/


/-- The module filter basis associated to a ring filter basis and a compatible submodule basis.
This allows to build a topological module structure compatible with the given module structure
and the topology associated to the given ring filter basis. -/
def RingFilterBasis.moduleFilterBasis [Nonempty ι] (BR : RingFilterBasis R) {B : ι → Submodule R M}
    (hB : BR.SubmodulesBasis B) : @ModuleFilterBasis R M _ BR.topology _ _ :=
  @SubmodulesBasis.toModuleFilterBasis ι R _ M _ _ BR.topology _ _ (BR.submodulesBasisIsBasis hB)

