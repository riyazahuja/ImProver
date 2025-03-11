/-- A Gδ set is a countable intersection of open sets. -/
def IsGδ (s : Set X) : Prop :=
  ∃ T : Set (Set X), (∀ t ∈ T, IsOpen t) ∧ T.Countable ∧ s = ⋂₀ T


/-- An open set is a Gδ set. -/
theorem IsOpen.isGδ {s : Set X} (h : IsOpen s) : IsGδ s :=
           /-
             X : Type u_1
             inst✝ : TopologicalSpace X
             s : Set X
             h : IsOpen s
             ⊢ ∀ (t : Set X), Membership.mem (Singleton.singleton s) t → IsOpen t
           -/
  ⟨{s}, by simp [h], countable_singleton _, (Set.sInter_singleton _).symm⟩
           /-
             🎉 no goals
           -/


@[simp]
protected theorem IsGδ.empty : IsGδ (∅ : Set X) :=
  isOpen_empty.isGδ


@[deprecated (since := "2024-02-15")] alias isGδ_empty := IsGδ.empty


@[simp]
protected theorem IsGδ.univ : IsGδ (univ : Set X) :=
  isOpen_univ.isGδ


@[deprecated (since := "2024-02-15")] alias isGδ_univ := IsGδ.univ


theorem IsGδ.biInter_of_isOpen {I : Set ι} (hI : I.Countable) {f : ι → Set X}
    (hf : ∀ i ∈ I, IsOpen (f i)) : IsGδ (⋂ i ∈ I, f i) :=
              /-
                X : Type u_1
                ι : Type u_3
                inst✝ : TopologicalSpace X
                I : Set ι
                hI : I.Countable
                f : ι → Set X
                hf : ∀ (i : ι), Membership.mem I i → IsOpen (f i)
                ⊢ ∀ (t : Set X), Membership.mem (Set.image f I) t → IsOpen t
              -/
              /-
                🎉 no goals
              -/
  ⟨f '' I, by rwa [forall_mem_image], hI.image _, by rw [sInter_image]⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


@[deprecated (since := "2024-02-15")] alias isGδ_biInter_of_isOpen := IsGδ.biInter_of_isOpen


theorem IsGδ.iInter_of_isOpen [Countable ι'] {f : ι' → Set X} (hf : ∀ i, IsOpen (f i)) :
    IsGδ (⋂ i, f i) :=
               /-
                 X : Type u_1
                 ι' : Sort u_4
                 inst✝¹ : TopologicalSpace X
                 inst✝ : Countable ι'
                 f : ι' → Set X
                 hf : ∀ (i : ι'), IsOpen (f i)
                 ⊢ ∀ (t : Set X), Membership.mem (Set.range f) t → IsOpen t
               -/
               /-
                 🎉 no goals
               -/
  ⟨range f, by rwa [forall_mem_range], countable_range _, by rw [sInter_range]⟩
                                                             /-
                                                               🎉 no goals
                                                             -/


@[deprecated (since := "2024-02-15")] alias isGδ_iInter_of_isOpen := IsGδ.iInter_of_isOpen


lemma isGδ_iff_eq_iInter_nat {s : Set X} :
    IsGδ s ↔ ∃ (f : ℕ → Set X), (∀ n, IsOpen (f n)) ∧ s = ⋂ n, f n := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsGδ s) (Exists fun f => And (∀ (n : Nat), IsOpen (f n)) (Eq s (Set.iIn …
  -/
  refine ⟨?_, ?_⟩
    /-
      case refine_1
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ IsGδ s → Exists fun f => And (∀ (n : Nat), IsOpen (f n)) (Eq s (Set.iInter f …
    -/
  · rintro ⟨T, hT, T_count, rfl⟩
    /-
      case refine_1.intro.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      T : Set (Set X)
      hT : ∀ (t : Set X), Membership.mem T t → IsOpen t
      T_count : T.Countable
      ⊢ Exists fun f => And (∀ (n : Nat), IsOpen (f n)) (Eq T.sInter (Set.iInter fun …
    -/
    rcases Set.eq_empty_or_nonempty T with rfl|hT
      /-
        case refine_1.intro.intro.intro.inl
        X : Type u_1
        inst✝ : TopologicalSpace X
        hT : ∀ (t : Set X), Membership.mem EmptyCollection.emptyCollection t → IsOpen t
        T_count : EmptyCollection.emptyCollection.Countable
        ⊢ Exists fun f => And (∀ (n : Nat), IsOpen (f n)) (Eq EmptyCollection.emptyCol …
      -/
    · exact ⟨fun _n ↦ univ, fun _n ↦ isOpen_univ, by simp⟩
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.inr
        X : Type u_1
        inst✝ : TopologicalSpace X
        T : Set (Set X)
        hT✝ : ∀ (t : Set X), Membership.mem T t → IsOpen t
        T_count : T.Countable
        hT : T.Nonempty
        ⊢ Exists fun f => And (∀ (n : Nat), IsOpen (f n)) (Eq T.sInter (Set.iInter fun …
      -/
    · obtain ⟨f, hf⟩ : ∃ (f : ℕ → Set X), T = range f := Countable.exists_eq_range T_count hT
      /-
        case refine_1.intro.intro.intro.inr.intro
        X : Type u_1
        inst✝ : TopologicalSpace X
        T : Set (Set X)
        hT✝ : ∀ (t : Set X), Membership.mem T t → IsOpen t
        T_count : T.Countable
        hT : T.Nonempty
        f : Nat → Set X
        hf : Eq T (Set.range f)
        ⊢ Exists fun f => And (∀ (n : Nat), IsOpen (f n)) (Eq T.sInter (Set.iInter fun …
      -/
      exact ⟨f, by aesop, by simp [hf]⟩
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      X : Type u_1
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ (Exists fun f => And (∀ (n : Nat), IsOpen (f n)) (Eq s (Set.iInter fun n =>  …
    -/
  · rintro ⟨f, hf, rfl⟩
    /-
      case refine_2.intro.intro
      X : Type u_1
      inst✝ : TopologicalSpace X
      f : Nat → Set X
      hf : ∀ (n : Nat), IsOpen (f n)
      ⊢ IsGδ (Set.iInter fun n => f n)
    -/
    exact .iInter_of_isOpen hf
    /-
      🎉 no goals
    -/


alias ⟨IsGδ.eq_iInter_nat, _⟩ := isGδ_iff_eq_iInter_nat


/-- The intersection of an encodable family of Gδ sets is a Gδ set. -/
protected theorem IsGδ.iInter [Countable ι'] {s : ι' → Set X} (hs : ∀ i, IsGδ (s i)) :
    IsGδ (⋂ i, s i) := by
  /-
    X : Type u_1
    ι' : Sort u_4
    inst✝¹ : TopologicalSpace X
    inst✝ : Countable ι'
    s : ι' → Set X
    hs : ∀ (i : ι'), IsGδ (s i)
    ⊢ IsGδ (Set.iInter fun i => s i)
  -/
  choose T hTo hTc hTs using hs
  /-
    X : Type u_1
    ι' : Sort u_4
    inst✝¹ : TopologicalSpace X
    inst✝ : Countable ι'
    s : ι' → Set X
    T : ι' → Set (Set X)
    hTo : ∀ (i : ι') (t : Set X), Membership.mem (T i) t → IsOpen t
    hTc : ∀ (i : ι'), (T i).Countable
    hTs : ∀ (i : ι'), Eq (s i) (T i).sInter
    ⊢ IsGδ (Set.iInter fun i => s i)
  -/
  obtain rfl : s = fun i => ⋂₀ T i := funext hTs
  /-
    X : Type u_1
    ι' : Sort u_4
    inst✝¹ : TopologicalSpace X
    inst✝ : Countable ι'
    T : ι' → Set (Set X)
    hTo : ∀ (i : ι') (t : Set X), Membership.mem (T i) t → IsOpen t
    hTc : ∀ (i : ι'), (T i).Countable
    hTs : ∀ (i : ι'), Eq ((fun i => (T i).sInter) i) (T i).sInter
    ⊢ IsGδ (Set.iInter fun i => (fun i => (T i).sInter) i)
  -/
  refine ⟨⋃ i, T i, ?_, countable_iUnion hTc, (sInter_iUnion _).symm⟩
  /-
    X : Type u_1
    ι' : Sort u_4
    inst✝¹ : TopologicalSpace X
    inst✝ : Countable ι'
    T : ι' → Set (Set X)
    hTo : ∀ (i : ι') (t : Set X), Membership.mem (T i) t → IsOpen t
    hTc : ∀ (i : ι'), (T i).Countable
    hTs : ∀ (i : ι'), Eq ((fun i => (T i).sInter) i) (T i).sInter
    ⊢ ∀ (t : Set X), Membership.mem (Set.iUnion fun i => T i) t → IsOpen t
  -/
  simpa [@forall_swap ι'] using hTo
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024.02.15")] alias isGδ_iInter := IsGδ.iInter


theorem IsGδ.biInter {s : Set ι} (hs : s.Countable) {t : ∀ i ∈ s, Set X}
    (ht : ∀ (i) (hi : i ∈ s), IsGδ (t i hi)) : IsGδ (⋂ i ∈ s, t i ‹_›) := by
  /-
    X : Type u_1
    ι : Type u_3
    inst✝ : TopologicalSpace X
    s : Set ι
    hs : s.Countable
    t : (i : ι) → Membership.mem s i → Set X
    ht : ∀ (i : ι) (hi : Membership.mem s i), IsGδ (t i hi)
    ⊢ IsGδ (Set.iInter fun i => Set.iInter fun h => t i h)
  -/
  rw [biInter_eq_iInter]
  /-
    X : Type u_1
    ι : Type u_3
    inst✝ : TopologicalSpace X
    s : Set ι
    hs : s.Countable
    t : (i : ι) → Membership.mem s i → Set X
    ht : ∀ (i : ι) (hi : Membership.mem s i), IsGδ (t i hi)
    ⊢ IsGδ (Set.iInter fun x => t ↑x ⋯)
  -/
  haveI := hs.to_subtype
  /-
    X : Type u_1
    ι : Type u_3
    inst✝ : TopologicalSpace X
    s : Set ι
    hs : s.Countable
    t : (i : ι) → Membership.mem s i → Set X
    ht : ∀ (i : ι) (hi : Membership.mem s i), IsGδ (t i hi)
    this : Countable ↑s
    ⊢ IsGδ (Set.iInter fun x => t ↑x ⋯)
  -/
  exact .iInter fun x => ht x x.2
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-15")] alias isGδ_biInter := IsGδ.biInter


/-- A countable intersection of Gδ sets is a Gδ set. -/
theorem IsGδ.sInter {S : Set (Set X)} (h : ∀ s ∈ S, IsGδ s) (hS : S.Countable) : IsGδ (⋂₀ S) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    h : ∀ (s : Set X), Membership.mem S s → IsGδ s
    hS : S.Countable
    ⊢ IsGδ S.sInter
  -/
  simpa only [sInter_eq_biInter] using IsGδ.biInter hS h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-15")] alias isGδ_sInter := IsGδ.sInter


theorem IsGδ.inter {s t : Set X} (hs : IsGδ s) (ht : IsGδ t) : IsGδ (s ∩ t) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsGδ s
    ht : IsGδ t
    ⊢ IsGδ (Inter.inter s t)
  -/
  rw [inter_eq_iInter]
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsGδ s
    ht : IsGδ t
    ⊢ IsGδ (Set.iInter fun b => cond b s t)
  -/
  exact .iInter (Bool.forall_bool.2 ⟨ht, hs⟩)
  /-
    🎉 no goals
  -/


/-- The union of two Gδ sets is a Gδ set. -/
theorem IsGδ.union {s t : Set X} (hs : IsGδ s) (ht : IsGδ t) : IsGδ (s ∪ t) := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s t : Set X
    hs : IsGδ s
    ht : IsGδ t
    ⊢ IsGδ (Union.union s t)
  -/
  rcases hs with ⟨S, Sopen, Scount, rfl⟩
  /-
    case intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    t : Set X
    ht : IsGδ t
    S : Set (Set X)
    Sopen : ∀ (t : Set X), Membership.mem S t → IsOpen t
    Scount : S.Countable
    ⊢ IsGδ (Union.union S.sInter t)
  -/
  rcases ht with ⟨T, Topen, Tcount, rfl⟩
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    Sopen : ∀ (t : Set X), Membership.mem S t → IsOpen t
    Scount : S.Countable
    T : Set (Set X)
    Topen : ∀ (t : Set X), Membership.mem T t → IsOpen t
    Tcount : T.Countable
    ⊢ IsGδ (Union.union S.sInter T.sInter)
  -/
  rw [sInter_union_sInter]
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    Sopen : ∀ (t : Set X), Membership.mem S t → IsOpen t
    Scount : S.Countable
    T : Set (Set X)
    Topen : ∀ (t : Set X), Membership.mem T t → IsOpen t
    Tcount : T.Countable
    ⊢ IsGδ (Set.iInter fun p => Set.iInter fun h => Union.union p.1 p.2)
  -/
  refine .biInter_of_isOpen (Scount.prod Tcount) ?_
  /-
    case intro.intro.intro.intro.intro.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    Sopen : ∀ (t : Set X), Membership.mem S t → IsOpen t
    Scount : S.Countable
    T : Set (Set X)
    Topen : ∀ (t : Set X), Membership.mem T t → IsOpen t
    Tcount : T.Countable
    ⊢ ∀ (i : Prod (Set X) (Set X)), Membership.mem (SProd.sprod S T) i → IsOpen (U …
  -/
  rintro ⟨a, b⟩ ⟨ha, hb⟩
  /-
    case intro.intro.intro.intro.intro.intro.mk.intro
    X : Type u_1
    inst✝ : TopologicalSpace X
    S : Set (Set X)
    Sopen : ∀ (t : Set X), Membership.mem S t → IsOpen t
    Scount : S.Countable
    T : Set (Set X)
    Topen : ∀ (t : Set X), Membership.mem T t → IsOpen t
    Tcount : T.Countable
    a b : Set X
    ha : Membership.mem S { fst := a, snd := b }.1
    hb : Membership.mem T { fst := a, snd := b }.2
    ⊢ IsOpen (Union.union { fst := a, snd := b }.1 { fst := a, snd := b }.2)
  -/
  exact (Sopen a ha).union (Topen b hb)
  /-
    🎉 no goals
  -/


/-- The union of finitely many Gδ sets is a Gδ set, `Set.sUnion` version. -/
theorem IsGδ.sUnion {S : Set (Set X)} (hS : S.Finite) (h : ∀ s ∈ S, IsGδ s) : IsGδ (⋃₀ S) := by
  induction S, hS using Set.Finite.dinduction_on with
  | H0 => simp
  | H1 _ _ ih =>
    simp only [forall_mem_insert, sUnion_insert] at *
    exact h.1.union (ih h.2)


/-- The union of finitely many Gδ sets is a Gδ set, bounded indexed union version. -/
theorem IsGδ.biUnion {s : Set ι} (hs : s.Finite) {f : ι → Set X} (h : ∀ i ∈ s, IsGδ (f i)) :
    IsGδ (⋃ i ∈ s, f i) := by
  /-
    X : Type u_1
    ι : Type u_3
    inst✝ : TopologicalSpace X
    s : Set ι
    hs : s.Finite
    f : ι → Set X
    h : ∀ (i : ι), Membership.mem s i → IsGδ (f i)
    ⊢ IsGδ (Set.iUnion fun i => Set.iUnion fun h => f i)
  -/
  rw [← sUnion_image]
  /-
    X : Type u_1
    ι : Type u_3
    inst✝ : TopologicalSpace X
    s : Set ι
    hs : s.Finite
    f : ι → Set X
    h : ∀ (i : ι), Membership.mem s i → IsGδ (f i)
    ⊢ IsGδ (Set.image f s).sUnion
  -/
  exact .sUnion (hs.image _) (forall_mem_image.2 h)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-15")]
alias isGδ_biUnion := IsGδ.biUnion


/-- The union of finitely many Gδ sets is a Gδ set, bounded indexed union version. -/
theorem IsGδ.iUnion [Finite ι'] {f : ι' → Set X} (h : ∀ i, IsGδ (f i)) : IsGδ (⋃ i, f i) :=
  .sUnion (finite_range _) <| forall_mem_range.2 h


/-- A set `s` is called *residual* if it includes a countable intersection of dense open sets. -/
def residual (X : Type*) [TopologicalSpace X] : Filter X :=
  Filter.countableGenerate { t | IsOpen t ∧ Dense t }


instance countableInterFilter_residual : CountableInterFilter (residual X) := by
  /-
    X : Type u_1
    Y : Type u_2
    ι : Type u_3
    ι' : Sort u_4
    inst✝ : TopologicalSpace X
    ⊢ CountableInterFilter (residual X)
  -/
  rw [residual]; infer_instance
                 /-
                   🎉 no goals
                 -/


/-- Dense open sets are residual. -/
theorem residual_of_dense_open {s : Set X} (ho : IsOpen s) (hd : Dense s) : s ∈ residual X :=
  CountableGenerateSets.basic ⟨ho, hd⟩


/-- Dense Gδ sets are residual. -/
theorem residual_of_dense_Gδ {s : Set X} (ho : IsGδ s) (hd : Dense s) : s ∈ residual X := by
  /-
    X : Type u_1
    inst✝ : TopologicalSpace X
    s : Set X
    ho : IsGδ s
    hd : Dense s
    ⊢ Membership.mem (residual X) s
  -/
  rcases ho with ⟨T, To, Tct, rfl⟩
  exact
    (countable_sInter_mem Tct).mpr fun t tT =>
      residual_of_dense_open (To t tT) (hd.mono (sInter_subset_of_mem tT))


/-- A set is residual iff it includes a countable intersection of dense open sets. -/
theorem mem_residual_iff {s : Set X} :
    s ∈ residual X ↔
      ∃ S : Set (Set X), (∀ t ∈ S, IsOpen t) ∧ (∀ t ∈ S, Dense t) ∧ S.Countable ∧ ⋂₀ S ⊆ s :=
                                        /-
                                          X : Type u_1
                                          inst✝ : TopologicalSpace X
                                          s : Set X
                                          ⊢ Iff (Exists fun S => And (HasSubset.Subset S (setOf fun t => And (IsOpen t)  …
                                        -/
  mem_countableGenerate_iff.trans <| by simp_rw [subset_def, mem_setOf, forall_and, and_assoc]
                                        /-
                                          🎉 no goals
                                        -/


/-- A set is called **nowhere dense** iff its closure has empty interior. -/
def IsNowhereDense (s : Set X) := interior (closure s) = ∅


/-- The empty set is nowhere dense. -/
@[simp]
lemma isNowhereDense_empty : IsNowhereDense (∅ : Set X) := by
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    ⊢ IsNowhereDense EmptyCollection.emptyCollection
  -/
  rw [IsNowhereDense, closure_empty, interior_empty]
  /-
    🎉 no goals
  -/


/-- A closed set is nowhere dense iff its interior is empty. -/
lemma IsClosed.isNowhereDense_iff {s : Set X} (hs : IsClosed s) :
    IsNowhereDense s ↔ interior s = ∅ := by
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsClosed s
    ⊢ Iff (IsNowhereDense s) (Eq (interior s) EmptyCollection.emptyCollection)
  -/
  rw [IsNowhereDense, IsClosed.closure_eq hs]
  /-
    🎉 no goals
  -/


/-- If a set `s` is nowhere dense, so is its closure. -/
protected lemma IsNowhereDense.closure {s : Set X} (hs : IsNowhereDense s) :
    IsNowhereDense (closure s) := by
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    s : Set X
    hs : IsNowhereDense s
    ⊢ IsNowhereDense (closure s)
  -/
  rwa [IsNowhereDense, closure_closure]
  /-
    🎉 no goals
  -/


/-- A nowhere dense set `s` is contained in a closed nowhere dense set (namely, its closure). -/
lemma IsNowhereDense.subset_of_closed_isNowhereDense {s : Set X} (hs : IsNowhereDense s) :
    ∃ t : Set X, s ⊆ t ∧ IsNowhereDense t ∧ IsClosed t :=
  ⟨closure s, subset_closure, ⟨hs.closure, isClosed_closure⟩⟩


/-- A set `s` is closed and nowhere dense iff its complement `sᶜ` is open and dense. -/
lemma isClosed_isNowhereDense_iff_compl {s : Set X} :
    IsClosed s ∧ IsNowhereDense s ↔ IsOpen sᶜ ∧ Dense sᶜ := by
  rw [and_congr_right IsClosed.isNowhereDense_iff,
    isOpen_compl_iff, interior_eq_empty_iff_dense_compl]


/-- A set is called **meagre** iff its complement is a residual (or comeagre) set. -/
def IsMeagre (s : Set X) := sᶜ ∈ residual X


/-- The empty set is meagre. -/
lemma meagre_empty : IsMeagre (∅ : Set X) := by
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    ⊢ IsMeagre EmptyCollection.emptyCollection
  -/
  rw [IsMeagre, compl_empty]
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    ⊢ Membership.mem (residual X) Set.univ
  -/
  exact Filter.univ_mem
  /-
    🎉 no goals
  -/


/-- Subsets of meagre sets are meagre. -/
lemma IsMeagre.mono {s t : Set X} (hs : IsMeagre s) (hts : t ⊆ s) : IsMeagre t :=
  Filter.mem_of_superset hs (compl_subset_compl.mpr hts)


/-- An intersection with a meagre set is meagre. -/
lemma IsMeagre.inter {s t : Set X} (hs : IsMeagre s) : IsMeagre (s ∩ t) :=
  hs.mono inter_subset_left


/-- A countable union of meagre sets is meagre. -/
lemma isMeagre_iUnion {s : ℕ → Set X} (hs : ∀ n, IsMeagre (s n)) : IsMeagre (⋃ n, s n) := by
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    s : Nat → Set X
    hs : ∀ (n : Nat), IsMeagre (s n)
    ⊢ IsMeagre (Set.iUnion fun n => s n)
  -/
  rw [IsMeagre, compl_iUnion]
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    s : Nat → Set X
    hs : ∀ (n : Nat), IsMeagre (s n)
    ⊢ Membership.mem (residual X) (Set.iInter fun i => HasCompl.compl (s i))
  -/
  exact countable_iInter_mem.mpr hs
  /-
    🎉 no goals
  -/


/-- A set is meagre iff it is contained in a countable union of nowhere dense sets. -/
lemma isMeagre_iff_countable_union_isNowhereDense {s : Set X} :
    IsMeagre s ↔ ∃ S : Set (Set X), (∀ t ∈ S, IsNowhereDense t) ∧ S.Countable ∧ s ⊆ ⋃₀ S := by
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (IsMeagre s) (Exists fun S => And (∀ (t : Set X), Membership.mem S t → I …
  -/
  rw [IsMeagre, mem_residual_iff, compl_bijective.surjective.image_surjective.exists]
  simp_rw [← and_assoc, ← forall_and, forall_mem_image, ← isClosed_isNowhereDense_iff_compl,
    sInter_image, ← compl_iUnion₂, compl_subset_compl, ← sUnion_eq_biUnion, and_assoc]
  /-
    X : Type u_5
    inst✝ : TopologicalSpace X
    s : Set X
    ⊢ Iff (Exists fun x => And (∀ ⦃x_1 : Set X⦄, Membership.mem x x_1 → And (IsClo …
  -/
  refine ⟨fun ⟨S, hS, hc, hsub⟩ ↦ ⟨S, fun s hs ↦ (hS hs).2, ?_, hsub⟩, ?_⟩
    /-
      case refine_1
      X : Type u_5
      inst✝ : TopologicalSpace X
      s : Set X
      x✝ : Exists fun x => And (∀ ⦃x_1 : Set X⦄, Membership.mem x x_1 → And (IsClose …
      S : Set (Set X)
      hS : ∀ ⦃x : Set X⦄, Membership.mem S x → And (IsClosed x) (IsNowhereDense x)
      hc : (Set.image HasCompl.compl S).Countable
      hsub : HasSubset.Subset s S.sUnion
      ⊢ S.Countable
    -/
  · rw [← compl_compl_image S]; exact hc.image _
                                /-
                                  🎉 no goals
                                -/
    /-
      case refine_2
      X : Type u_5
      inst✝ : TopologicalSpace X
      s : Set X
      ⊢ (Exists fun S => And (∀ (t : Set X), Membership.mem S t → IsNowhereDense t)  …
    -/
  · intro ⟨S, hS, hc, hsub⟩
    /-
      case refine_2
      X : Type u_5
      inst✝ : TopologicalSpace X
      s : Set X
      S : Set (Set X)
      hS : ∀ (t : Set X), Membership.mem S t → IsNowhereDense t
      hc : S.Countable
      hsub : HasSubset.Subset s S.sUnion
      ⊢ Exists fun x => And (∀ ⦃x_1 : Set X⦄, Membership.mem x x_1 → And (IsClosed x …
    -/
    use closure '' S
    /-
      case h
      X : Type u_5
      inst✝ : TopologicalSpace X
      s : Set X
      S : Set (Set X)
      hS : ∀ (t : Set X), Membership.mem S t → IsNowhereDense t
      hc : S.Countable
      hsub : HasSubset.Subset s S.sUnion
      ⊢ And (∀ ⦃x : Set X⦄, Membership.mem (Set.image closure S) x → And (IsClosed x …
    -/
    rw [forall_mem_image]
    exact ⟨fun s hs ↦ ⟨isClosed_closure, (hS s hs).closure⟩,
      (hc.image _).image _, hsub.trans (sUnion_mono_subsets fun s ↦ subset_closure)⟩


