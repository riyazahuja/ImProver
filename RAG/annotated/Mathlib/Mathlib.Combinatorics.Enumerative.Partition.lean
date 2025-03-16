/-- A partition of `n` is a multiset of positive integers summing to `n`. -/
@[ext]
structure Partition (n : ℕ) where
  /-- positive integers summing to `n`-/
  parts : Multiset ℕ
  /-- proof that the `parts` are positive -/
  parts_pos : ∀ {i}, i ∈ parts → 0 < i
  /-- proof that the `parts` sum to `n`-/
  parts_sum : parts.sum = n
  -- Porting note: chokes on `parts_pos`
  --deriving DecidableEq


instance decidableEqPartition {n : ℕ} : DecidableEq (Partition n) :=
  fun _ _ => decidable_of_iff' _ Partition.ext_iff


/-- A composition induces a partition (just convert the list to a multiset). -/
@[simps]
def ofComposition (n : ℕ) (c : Composition n) : Partition n where
  parts := c.blocks
  parts_pos hi := c.blocks_pos hi
                  /-
                    n : Nat
                    c : Composition n
                    ⊢ Eq (↑c.blocks).sum n
                  -/
  parts_sum := by rw [Multiset.sum_coe, c.blocks_sum]
                  /-
                    🎉 no goals
                  -/


theorem ofComposition_surj {n : ℕ} : Function.Surjective (ofComposition n) := by
  /-
    n : Nat
    ⊢ Function.Surjective (Nat.Partition.ofComposition n)
  -/
  rintro ⟨b, hb₁, hb₂⟩
  /-
    case mk
    n : Nat
    b : Multiset Nat
    hb₁ : ∀ {i : Nat}, Membership.mem b i → LT.lt 0 i
    hb₂ : Eq b.sum n
    ⊢ Exists fun a => Eq (Nat.Partition.ofComposition n a) { parts := b, parts_pos …
  -/
  induction b using Quotient.inductionOn with | _ b => ?_
  /-
    case mk.h
    n : Nat
    b : List Nat
    hb₁ : ∀ {i : Nat}, Membership.mem (Quotient.mk (List.isSetoid Nat) b) i → LT.l …
    hb₂ : Eq (Multiset.sum (Quotient.mk (List.isSetoid Nat) b)) n
    ⊢ Exists fun a => Eq (Nat.Partition.ofComposition n a) { parts := Quotient.mk  …
  -/
  exact ⟨⟨b, hb₁, by simpa using hb₂⟩, Partition.ext rfl⟩
  /-
    🎉 no goals
  -/

-- The argument `n` is kept explicit here since it is useful in tactic mode proofs to generate the
-- proof obligation `l.sum = n`.

/-- Given a multiset which sums to `n`, construct a partition of `n` with the same multiset, but
without the zeros.
-/
@[simps]
def ofSums (n : ℕ) (l : Multiset ℕ) (hl : l.sum = n) : Partition n where
  parts := l.filter (· ≠ 0)
  parts_pos hi := (of_mem_filter hi).bot_lt
  parts_sum := by
    /-
      n : Nat
      l : Multiset Nat
      hl : Eq l.sum n
      ⊢ Eq (Multiset.filter (fun x => Ne x 0) l).sum n
    -/
    have lz : (l.filter (· = 0)).sum = 0 := by simp [sum_eq_zero_iff]
    /-
      n : Nat
      l : Multiset Nat
      hl : Eq l.sum n
      lz : Eq (Multiset.filter (fun x => Eq x 0) l).sum 0
      ⊢ Eq (Multiset.filter (fun x => Ne x 0) l).sum n
    -/
    rwa [← filter_add_not (· = 0) l, sum_add, lz, zero_add] at hl
    /-
      🎉 no goals
    -/


/-- A `Multiset ℕ` induces a partition on its sum. -/
@[simps!]
def ofMultiset (l : Multiset ℕ) : Partition l.sum := ofSums _ l rfl


/-- An element `s` of `Sym σ n` induces a partition given by its multiplicities. -/
def ofSym {n : ℕ} {σ : Type*} (s : Sym σ n) [DecidableEq σ] : n.Partition where
  parts := s.1.dedup.map s.1.count
                  /-
                    n : Nat
                    σ : Type u_1
                    s : Sym σ n
                    inst✝ : DecidableEq σ
                    ⊢ ∀ {i : Nat}, Membership.mem (Multiset.map (fun a => Multiset.count a ↑s) (↑s …
                  -/
  parts_pos := by simp [Multiset.count_pos]
                  /-
                    🎉 no goals
                  -/
  parts_sum := by
    /-
      n : Nat
      σ : Type u_1
      s : Sym σ n
      inst✝ : DecidableEq σ
      ⊢ Eq (Multiset.map (fun a => Multiset.count a ↑s) (↑s).dedup).sum n
    -/
    show ∑ a ∈ s.1.toFinset, count a s.1 = n
    /-
      n : Nat
      σ : Type u_1
      s : Sym σ n
      inst✝ : DecidableEq σ
      ⊢ Eq ((↑s).toFinset.sum fun a => Multiset.count a ↑s) n
    -/
    rw [toFinset_sum_count_eq]
    /-
      n : Nat
      σ : Type u_1
      s : Sym σ n
      inst✝ : DecidableEq σ
      ⊢ Eq (↑s).card n
    -/
    exact s.2
    /-
      🎉 no goals
    -/


@[simp] lemma ofSym_map (e : σ ≃ τ) (s : Sym σ n) :
    ofSym (s.map e) = ofSym s := by
  /-
    n : Nat
    σ : Type u_1
    τ : Type u_2
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    e : Equiv σ τ
    s : Sym σ n
    ⊢ Eq (Nat.Partition.ofSym (Sym.map (⇑e) s)) (Nat.Partition.ofSym s)
  -/
  simp only [ofSym, Sym.val_eq_coe, Sym.coe_map, toFinset_val, mk.injEq]
  /-
    n : Nat
    σ : Type u_1
    τ : Type u_2
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    e : Equiv σ τ
    s : Sym σ n
    ⊢ Eq (Multiset.map (fun x => Multiset.count x (Multiset.map ⇑e ↑s)) (Multiset. …
  -/
  rw [Multiset.dedup_map_of_injective e.injective]
  /-
    n : Nat
    σ : Type u_1
    τ : Type u_2
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    e : Equiv σ τ
    s : Sym σ n
    ⊢ Eq (Multiset.map (fun x => Multiset.count x (Multiset.map ⇑e ↑s)) (Multiset. …
  -/
  simp only [map_map, Function.comp_apply]
  /-
    n : Nat
    σ : Type u_1
    τ : Type u_2
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    e : Equiv σ τ
    s : Sym σ n
    ⊢ Eq (Multiset.map (fun x => Multiset.count (e x) (Multiset.map ⇑e ↑s)) (↑s).d …
  -/
  congr; funext i
  /-
    case e_f.h
    n : Nat
    σ : Type u_1
    τ : Type u_2
    inst✝¹ : DecidableEq σ
    inst✝ : DecidableEq τ
    e : Equiv σ τ
    s : Sym σ n
    i : σ
    ⊢ Eq (Multiset.count (e i) (Multiset.map ⇑e ↑s)) (Multiset.count i ↑s)
  -/
  rw [← Multiset.count_map_eq_count' e _ e.injective]
  /-
    🎉 no goals
  -/


/-- An equivalence between `σ` and `τ` induces an equivalence between the subtypes of `Sym σ n` and
`Sym τ n` corresponding to a given partition. -/
def ofSymShapeEquiv (μ : Partition n) (e : σ ≃ τ) :
    {x : Sym σ n // ofSym x = μ} ≃ {x : Sym τ n // ofSym x = μ} where
                                            /-
                                              n : Nat
                                              σ : Type u_1
                                              τ : Type u_2
                                              inst✝¹ : DecidableEq σ
                                              inst✝ : DecidableEq τ
                                              μ : n.Partition
                                              e : Equiv σ τ
                                              x : Subtype fun x => Eq (Nat.Partition.ofSym x) μ
                                              ⊢ Eq (Nat.Partition.ofSym ((Sym.equivCongr e) ↑x)) μ
                                            -/
  toFun := fun x => ⟨Sym.equivCongr e x, by simp [ofSym_map, x.2]⟩
                                            /-
                                              🎉 no goals
                                            -/
                                                  /-
                                                    n : Nat
                                                    σ : Type u_1
                                                    τ : Type u_2
                                                    inst✝¹ : DecidableEq σ
                                                    inst✝ : DecidableEq τ
                                                    μ : n.Partition
                                                    e : Equiv σ τ
                                                    x : Subtype fun x => Eq (Nat.Partition.ofSym x) μ
                                                    ⊢ Eq (Nat.Partition.ofSym ((Sym.equivCongr e.symm) ↑x)) μ
                                                  -/
  invFun := fun x => ⟨Sym.equivCongr e.symm x, by simp [ofSym_map, x.2]⟩
                                                  /-
                                                    🎉 no goals
                                                  -/
                 /-
                   n : Nat
                   σ : Type u_1
                   τ : Type u_2
                   inst✝¹ : DecidableEq σ
                   inst✝ : DecidableEq τ
                   μ : n.Partition
                   e : Equiv σ τ
                   ⊢ Function.LeftInverse (fun x => ⟨(Sym.equivCongr e.symm) ↑x, ⋯⟩) fun x => ⟨(S …
                 -/
  left_inv := by intro x; simp
                          /-
                            🎉 no goals
                          -/
                  /-
                    n : Nat
                    σ : Type u_1
                    τ : Type u_2
                    inst✝¹ : DecidableEq σ
                    inst✝ : DecidableEq τ
                    μ : n.Partition
                    e : Equiv σ τ
                    ⊢ Function.RightInverse (fun x => ⟨(Sym.equivCongr e.symm) ↑x, ⋯⟩) fun x => ⟨( …
                  -/
  right_inv := by intro x; simp
                           /-
                             🎉 no goals
                           -/


/-- The partition of exactly one part. -/
def indiscrete (n : ℕ) : Partition n := ofSums n {n} rfl


instance {n : ℕ} : Inhabited (Partition n) := ⟨indiscrete n⟩


@[simp] lemma indiscrete_parts {n : ℕ} (hn : n ≠ 0) : (indiscrete n).parts = {n} := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq (Nat.Partition.indiscrete n).parts (Singleton.singleton n)
  -/
  simp [indiscrete, filter_eq_self, hn]
  /-
    🎉 no goals
  -/


@[simp] lemma partition_zero_parts (p : Partition 0) : p.parts = 0 :=
  eq_zero_of_forall_not_mem fun _ h => (p.parts_pos h).ne' <| sum_eq_zero_iff.1 p.parts_sum _ h


instance UniquePartitionZero : Unique (Partition 0) where
                                /-
                                  n : Nat
                                  σ : Type u_1
                                  τ : Type u_2
                                  inst✝¹ : DecidableEq σ
                                  inst✝ : DecidableEq τ
                                  x✝ : Nat.Partition 0
                                  ⊢ Eq x✝.parts Inhabited.default.parts
                                -/
  uniq _ := Partition.ext <| by simp
                                /-
                                  🎉 no goals
                                -/


@[simp] lemma partition_one_parts (p : Partition 1) : p.parts = {1} := by
  have h : p.parts = replicate (card p.parts) 1 := eq_replicate_card.2 fun x hx =>
    ((le_sum_of_mem hx).trans_eq p.parts_sum).antisymm (p.parts_pos hx)
  /-
    p : Nat.Partition 1
    h : Eq p.parts (Multiset.replicate p.parts.card 1)
    ⊢ Eq p.parts (Singleton.singleton 1)
  -/
  have h' : card p.parts = 1 := by simpa using (congrArg sum h.symm).trans p.parts_sum
  /-
    p : Nat.Partition 1
    h : Eq p.parts (Multiset.replicate p.parts.card 1)
    h' : Eq p.parts.card 1
    ⊢ Eq p.parts (Singleton.singleton 1)
  -/
  rw [h, h', replicate_one]
  /-
    🎉 no goals
  -/


instance UniquePartitionOne : Unique (Partition 1) where
                                /-
                                  n : Nat
                                  σ : Type u_1
                                  τ : Type u_2
                                  inst✝¹ : DecidableEq σ
                                  inst✝ : DecidableEq τ
                                  x✝ : Nat.Partition 1
                                  ⊢ Eq x✝.parts Inhabited.default.parts
                                -/
  uniq _ := Partition.ext <| by simp
                                /-
                                  🎉 no goals
                                -/


@[simp] lemma ofSym_one (s : Sym σ 1) : ofSym s = indiscrete 1 := by
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    s : Sym σ 1
    ⊢ Eq (Nat.Partition.ofSym s) (Nat.Partition.indiscrete 1)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- The number of times a positive integer `i` appears in the partition `ofSums n l hl` is the same
as the number of times it appears in the multiset `l`.
(For `i = 0`, `Partition.non_zero` combined with `Multiset.count_eq_zero_of_not_mem` gives that
this is `0` instead.)
-/
theorem count_ofSums_of_ne_zero {n : ℕ} {l : Multiset ℕ} (hl : l.sum = n) {i : ℕ} (hi : i ≠ 0) :
    (ofSums n l hl).parts.count i = l.count i :=
  count_filter_of_pos hi


theorem count_ofSums_zero {n : ℕ} {l : Multiset ℕ} (hl : l.sum = n) :
    (ofSums n l hl).parts.count 0 = 0 :=
  count_filter_of_neg fun h => h rfl


/-- Show there are finitely many partitions by considering the surjection from compositions to
partitions.
-/
instance (n : ℕ) : Fintype (Partition n) :=
  Fintype.ofSurjective (ofComposition n) ofComposition_surj


/-- The finset of those partitions in which every part is odd. -/
def odds (n : ℕ) : Finset (Partition n) :=
  Finset.univ.filter fun c => ∀ i ∈ c.parts, ¬Even i


/-- The finset of those partitions in which each part is used at most once. -/
def distincts (n : ℕ) : Finset (Partition n) :=
  Finset.univ.filter fun c => c.parts.Nodup


/-- The finset of those partitions in which every part is odd and used at most once. -/
def oddDistincts (n : ℕ) : Finset (Partition n) :=
  odds n ∩ distincts n


