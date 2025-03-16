/-- The monotone sequence whose value at `n` is the supremum of the `f m` where `m ≤ n`. -/
def partialSups (f : ℕ → α) : ℕ →o α :=
  ⟨@Nat.rec (fun _ => α) (f 0) fun (n : ℕ) (a : α) => a ⊔ f (n + 1),
    monotone_nat_of_le_succ fun _ => le_sup_left⟩


@[simp]
theorem partialSups_zero (f : ℕ → α) : partialSups f 0 = f 0 :=
  rfl


@[simp]
theorem partialSups_succ (f : ℕ → α) (n : ℕ) :
    partialSups f (n + 1) = partialSups f n ⊔ f (n + 1) :=
  rfl


lemma partialSups_iff_forall {f : ℕ → α} (p : α → Prop)
    (hp : ∀ {a b}, p (a ⊔ b) ↔ p a ∧ p b) : ∀ {n : ℕ}, p (partialSups f n) ↔ ∀ k ≤ n, p (f k)
            /-
              α : Type u_1
              inst✝ : SemilatticeSup α
              f : Nat → α
              p : α → Prop
              hp : ∀ {a b : α}, Iff (p (Max.max a b)) (And (p a) (p b))
              ⊢ Iff (p ((partialSups f) 0)) (∀ (k : Nat), LE.le k 0 → p (f k))
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                  /-
                    α : Type u_1
                    inst✝ : SemilatticeSup α
                    f : Nat → α
                    p : α → Prop
                    hp : ∀ {a b : α}, Iff (p (Max.max a b)) (And (p a) (p b))
                    n : Nat
                    ⊢ Iff (p ((partialSups f) (HAdd.hAdd n 1))) (∀ (k : Nat), LE.le k (HAdd.hAdd n …
                  -/
  | (n + 1) => by simp [hp, partialSups_iff_forall, ← Nat.lt_succ_iff, ← Nat.forall_lt_succ]
                  /-
                    🎉 no goals
                  -/


@[simp]
lemma partialSups_le_iff {f : ℕ → α} {n : ℕ} {a : α} : partialSups f n ≤ a ↔ ∀ k ≤ n, f k ≤ a :=
  partialSups_iff_forall (· ≤ a) sup_le_iff


theorem le_partialSups_of_le (f : ℕ → α) {m n : ℕ} (h : m ≤ n) : f m ≤ partialSups f n :=
  partialSups_le_iff.1 le_rfl m h


theorem le_partialSups (f : ℕ → α) : f ≤ partialSups f := fun _n => le_partialSups_of_le f le_rfl


theorem partialSups_le (f : ℕ → α) (n : ℕ) (a : α) (w : ∀ m, m ≤ n → f m ≤ a) :
    partialSups f n ≤ a :=
  partialSups_le_iff.2 w


@[simp]
lemma upperBounds_range_partialSups (f : ℕ → α) :
    upperBounds (Set.range (partialSups f)) = upperBounds (Set.range f) := by
  /-
    α : Type u_1
    inst✝ : SemilatticeSup α
    f : Nat → α
    ⊢ Eq (upperBounds (Set.range ⇑(partialSups f))) (upperBounds (Set.range f))
  -/
  ext a
  /-
    case h
    α : Type u_1
    inst✝ : SemilatticeSup α
    f : Nat → α
    a : α
    ⊢ Iff (Membership.mem (upperBounds (Set.range ⇑(partialSups f))) a) (Membershi …
  -/
  simp only [mem_upperBounds, Set.forall_mem_range, partialSups_le_iff]
  /-
    case h
    α : Type u_1
    inst✝ : SemilatticeSup α
    f : Nat → α
    a : α
    ⊢ Iff (∀ (i k : Nat), LE.le k i → LE.le (f k) a) (∀ (i : Nat), LE.le (f i) a)
  -/
  exact ⟨fun h _ ↦ h _ _ le_rfl, fun h _ _ _ ↦ h _⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem bddAbove_range_partialSups {f : ℕ → α} :
    BddAbove (Set.range (partialSups f)) ↔ BddAbove (Set.range f) :=
  .of_eq <| congr_arg Set.Nonempty <| upperBounds_range_partialSups f


theorem Monotone.partialSups_eq {f : ℕ → α} (hf : Monotone f) : (partialSups f : ℕ → α) = f := by
  /-
    α : Type u_1
    inst✝ : SemilatticeSup α
    f : Nat → α
    hf : Monotone f
    ⊢ Eq (⇑(partialSups f)) f
  -/
  ext n
  /-
    case h
    α : Type u_1
    inst✝ : SemilatticeSup α
    f : Nat → α
    hf : Monotone f
    n : Nat
    ⊢ Eq ((partialSups f) n) (f n)
  -/
  induction' n with n ih
    /-
      case h.zero
      α : Type u_1
      inst✝ : SemilatticeSup α
      f : Nat → α
      hf : Monotone f
      ⊢ Eq ((partialSups f) 0) (f 0)
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      α : Type u_1
      inst✝ : SemilatticeSup α
      f : Nat → α
      hf : Monotone f
      n : Nat
      ih : Eq ((partialSups f) n) (f n)
      ⊢ Eq ((partialSups f) (HAdd.hAdd n 1)) (f (HAdd.hAdd n 1))
    -/
  · rw [partialSups_succ, ih, sup_eq_right.2 (hf (Nat.le_succ _))]
    /-
      🎉 no goals
    -/


theorem partialSups_mono : Monotone (partialSups : (ℕ → α) → ℕ →o α) := fun _f _g h _n ↦
  partialSups_le_iff.2 fun k hk ↦ (h k).trans (le_partialSups_of_le _ hk)


lemma partialSups_monotone (f : ℕ → α) : Monotone (partialSups f) :=
  fun n _ hnm ↦ partialSups_le f n _ (fun _ hm'n ↦ le_partialSups_of_le _ (hm'n.trans hnm))


/-- `partialSups` forms a Galois insertion with the coercion from monotone functions to functions.
-/
def partialSups.gi : GaloisInsertion (partialSups : (ℕ → α) → ℕ →o α) (↑) where
  choice f h :=
           /-
             α : Type u_1
             inst✝ : SemilatticeSup α
             f : Nat → α
             h : LE.le (⇑(partialSups f)) f
             ⊢ Monotone f
           -/
    ⟨f, by convert (partialSups f).monotone using 1; exact (le_partialSups f).antisymm h⟩
                                                     /-
                                                       🎉 no goals
                                                     -/
  gc f g := by
    /-
      α : Type u_1
      inst✝ : SemilatticeSup α
      f : Nat → α
      g : OrderHom Nat α
      ⊢ Iff (LE.le (partialSups f) g) (LE.le f ⇑g)
    -/
    refine ⟨(le_partialSups f).trans, fun h => ?_⟩
    /-
      α : Type u_1
      inst✝ : SemilatticeSup α
      f : Nat → α
      g : OrderHom Nat α
      h : LE.le f ⇑g
      ⊢ LE.le (partialSups f) g
    -/
    convert partialSups_mono h
    /-
      case h.e'_4
      α : Type u_1
      inst✝ : SemilatticeSup α
      f : Nat → α
      g : OrderHom Nat α
      h : LE.le f ⇑g
      ⊢ Eq g (partialSups ⇑g)
    -/
    exact OrderHom.ext _ _ g.monotone.partialSups_eq.symm
    /-
      🎉 no goals
    -/
  le_l_u f := le_partialSups f
  choice_eq f h := OrderHom.ext _ _ ((le_partialSups f).antisymm h)


theorem partialSups_eq_sup'_range (f : ℕ → α) (n : ℕ) :
    partialSups f n = (Finset.range (n + 1)).sup' ⟨n, Finset.self_mem_range_succ n⟩ f :=
                                 /-
                                   α : Type u_1
                                   inst✝ : SemilatticeSup α
                                   f : Nat → α
                                   n : Nat
                                   x✝ : α
                                   ⊢ Iff (LE.le ((partialSups f) n) x✝) (LE.le ((Finset.range (HAdd.hAdd n 1)).su …
                                 -/
  eq_of_forall_ge_iff fun _ ↦ by simp [Nat.lt_succ_iff]
                                 /-
                                   🎉 no goals
                                 -/


lemma partialSups_apply {ι : Type*} {π : ι → Type*} [(i : ι) → SemilatticeSup (π i)]
    (f : ℕ → (i : ι) → π i) (n : ℕ) (i : ι) : partialSups f n i = partialSups (f · i) n := by
  /-
    ι : Type u_2
    π : ι → Type u_3
    inst✝ : (i : ι) → SemilatticeSup (π i)
    f : Nat → (i : ι) → π i
    n : Nat
    i : ι
    ⊢ Eq ((partialSups f) n i) ((partialSups fun x => f x i) n)
  -/
  simp only [partialSups_eq_sup'_range, Finset.sup'_apply]
  /-
    🎉 no goals
  -/


theorem partialSups_eq_sup_range [SemilatticeSup α] [OrderBot α] (f : ℕ → α) (n : ℕ) :
    partialSups f n = (Finset.range (n + 1)).sup f :=
                                 /-
                                   α : Type u_1
                                   inst✝¹ : SemilatticeSup α
                                   inst✝ : OrderBot α
                                   f : Nat → α
                                   n : Nat
                                   x✝ : α
                                   ⊢ Iff (LE.le ((partialSups f) n) x✝) (LE.le ((Finset.range (HAdd.hAdd n 1)).su …
                                 -/
  eq_of_forall_ge_iff fun _ ↦ by simp [Nat.lt_succ_iff]
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
lemma disjoint_partialSups_left [DistribLattice α] [OrderBot α] {f : ℕ → α} {n : ℕ} {x : α} :
    Disjoint (partialSups f n) x ↔ ∀ k ≤ n, Disjoint (f k) x :=
  partialSups_iff_forall (Disjoint · x) disjoint_sup_left


@[simp]
lemma disjoint_partialSups_right [DistribLattice α] [OrderBot α] {f : ℕ → α} {n : ℕ} {x : α} :
    Disjoint x (partialSups f n) ↔ ∀ k ≤ n, Disjoint x (f k) :=
  partialSups_iff_forall (Disjoint x) disjoint_sup_right

/- Note this lemma requires a distributive lattice, so is not useful (or true) in situations such as
submodules. -/

theorem partialSups_disjoint_of_disjoint [DistribLattice α] [OrderBot α] (f : ℕ → α)
    (h : Pairwise (Disjoint on f)) {m n : ℕ} (hmn : m < n) : Disjoint (partialSups f m) (f n) :=
  disjoint_partialSups_left.2 fun _k hk ↦ h <| (hk.trans_lt hmn).ne


theorem partialSups_eq_ciSup_Iic (f : ℕ → α) (n : ℕ) : partialSups f n = ⨆ i : Set.Iic n, f i :=
  eq_of_forall_ge_iff fun _ ↦ by
    rw [ciSup_set_le_iff Set.nonempty_Iic ((Set.finite_le_nat _).image _).bddAbove,
                           /-
                             α : Type u_1
                             inst✝ : ConditionallyCompleteLattice α
                             f : Nat → α
                             n : Nat
                             x✝ : α
                             ⊢ Iff (∀ (k : Nat), LE.le k n → LE.le (f k) x✝) (∀ (i : Nat), Membership.mem ( …
                           -/
      partialSups_le_iff]; rfl
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem ciSup_partialSups_eq {f : ℕ → α} (h : BddAbove (Set.range f)) :
    ⨆ n, partialSups f n = ⨆ n, f n := by
  /-
    α : Type u_1
    inst✝ : ConditionallyCompleteLattice α
    f : Nat → α
    h : BddAbove (Set.range f)
    ⊢ Eq (iSup fun n => (partialSups f) n) (iSup fun n => f n)
  -/
  refine (ciSup_le fun n => ?_).antisymm (ciSup_mono ?_ <| le_partialSups f)
    /-
      case refine_1
      α : Type u_1
      inst✝ : ConditionallyCompleteLattice α
      f : Nat → α
      h : BddAbove (Set.range f)
      n : Nat
      ⊢ LE.le ((partialSups f) n) (iSup fun n => f n)
    -/
  · rw [partialSups_eq_ciSup_Iic]
    /-
      case refine_1
      α : Type u_1
      inst✝ : ConditionallyCompleteLattice α
      f : Nat → α
      h : BddAbove (Set.range f)
      n : Nat
      ⊢ LE.le (iSup fun i => f ↑i) (iSup fun n => f n)
    -/
    exact ciSup_le fun i => le_ciSup h _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      inst✝ : ConditionallyCompleteLattice α
      f : Nat → α
      h : BddAbove (Set.range f)
      ⊢ BddAbove (Set.range fun n => (partialSups f) n)
    -/
  · rwa [bddAbove_range_partialSups]
    /-
      🎉 no goals
    -/


theorem partialSups_eq_biSup [CompleteLattice α] (f : ℕ → α) (n : ℕ) :
    partialSups f n = ⨆ i ≤ n, f i := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f : Nat → α
    n : Nat
    ⊢ Eq ((partialSups f) n) (iSup fun i => iSup fun h => f i)
  -/
  simpa only [iSup_subtype] using partialSups_eq_ciSup_Iic f n
  /-
    🎉 no goals
  -/


lemma partialSups_eq_sUnion_image [DecidableEq (Set α)] (s : ℕ → Set α) (n : ℕ) :
    partialSups s n = ⋃₀ ↑((Finset.range (n + 1)).image s) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq (Set α)
    s : Nat → Set α
    n : Nat
    ⊢ Eq ((partialSups s) n) (↑(Finset.image s (Finset.range (HAdd.hAdd n 1)))).sU …
  -/
  ext; simp [partialSups_eq_biSup, Nat.lt_succ_iff]
       /-
         🎉 no goals
       -/


lemma partialSups_eq_biUnion_range (s : ℕ → Set α) (n : ℕ) :
    partialSups s n = ⋃ i ∈ Finset.range (n + 1), s i := by
  /-
    α : Type u_1
    s : Nat → Set α
    n : Nat
    ⊢ Eq ((partialSups s) n) (Set.iUnion fun i => Set.iUnion fun h => s i)
  -/
  ext; simp [partialSups_eq_biSup, Nat.lt_succ]
       /-
         🎉 no goals
       -/


theorem iSup_partialSups_eq (f : ℕ → α) : ⨆ n, partialSups f n = ⨆ n, f n :=
  ciSup_partialSups_eq <| OrderTop.bddAbove _


theorem iSup_le_iSup_of_partialSups_le_partialSups {f g : ℕ → α}
    (h : partialSups f ≤ partialSups g) : ⨆ n, f n ≤ ⨆ n, g n := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f g : Nat → α
    h : LE.le (partialSups f) (partialSups g)
    ⊢ LE.le (iSup fun n => f n) (iSup fun n => g n)
  -/
  rw [← iSup_partialSups_eq f, ← iSup_partialSups_eq g]
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f g : Nat → α
    h : LE.le (partialSups f) (partialSups g)
    ⊢ LE.le (iSup fun n => (partialSups f) n) (iSup fun n => (partialSups g) n)
  -/
  exact iSup_mono h
  /-
    🎉 no goals
  -/


theorem iSup_eq_iSup_of_partialSups_eq_partialSups {f g : ℕ → α}
    (h : partialSups f = partialSups g) : ⨆ n, f n = ⨆ n, g n := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    f g : Nat → α
    h : Eq (partialSups f) (partialSups g)
    ⊢ Eq (iSup fun n => f n) (iSup fun n => g n)
  -/
  simp_rw [← iSup_partialSups_eq f, ← iSup_partialSups_eq g, h]
  /-
    🎉 no goals
  -/


