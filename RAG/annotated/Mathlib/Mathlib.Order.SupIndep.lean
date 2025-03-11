/-- Supremum independence of finite sets. We avoid the "obvious" definition using `s.erase i`
because `erase` would require decidable equality on `ι`. -/
def SupIndep (s : Finset ι) (f : ι → α) : Prop :=
  ∀ ⦃t⦄, t ⊆ s → ∀ ⦃i⦄, i ∈ s → i ∉ t → Disjoint (f i) (t.sup f)


instance [DecidableEq ι] [DecidableEq α] : Decidable (SupIndep s f) := by
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    ι' : Type u_4
    inst✝³ : Lattice α
    inst✝² : OrderBot α
    s t : Finset ι
    f : ι → α
    i : ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq α
    ⊢ Decidable (s.SupIndep f)
  -/
  refine @Finset.decidableForallOfDecidableSubsets _ _ _ (?_)
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    ι' : Type u_4
    inst✝³ : Lattice α
    inst✝² : OrderBot α
    s t : Finset ι
    f : ι → α
    i : ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq α
    ⊢ (t : Finset ι) → HasSubset.Subset t s → Decidable (∀ ⦃i : ι⦄, Membership.mem …
  -/
  rintro t -
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    ι' : Type u_4
    inst✝³ : Lattice α
    inst✝² : OrderBot α
    s t✝ : Finset ι
    f : ι → α
    i : ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq α
    t : Finset ι
    ⊢ Decidable (∀ ⦃i : ι⦄, Membership.mem s i → Not (Membership.mem t i) → Disjoi …
  -/
  refine @Finset.decidableDforallFinset _ _ _ (?_)
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    ι' : Type u_4
    inst✝³ : Lattice α
    inst✝² : OrderBot α
    s t✝ : Finset ι
    f : ι → α
    i : ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq α
    t : Finset ι
    ⊢ (a : ι) → Membership.mem s a → Decidable (Not (Membership.mem t a) → Disjoin …
  -/
  rintro i -
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    ι' : Type u_4
    inst✝³ : Lattice α
    inst✝² : OrderBot α
    s t✝ : Finset ι
    f : ι → α
    i✝ : ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq α
    t : Finset ι
    i : ι
    ⊢ Decidable (Not (Membership.mem t i) → Disjoint (f i) (t.sup f))
  -/
  have : Decidable (Disjoint (f i) (sup t f)) := decidable_of_iff' (_ = ⊥) disjoint_iff
  /-
    α : Type u_1
    β : Type u_2
    ι : Type u_3
    ι' : Type u_4
    inst✝³ : Lattice α
    inst✝² : OrderBot α
    s t✝ : Finset ι
    f : ι → α
    i✝ : ι
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq α
    t : Finset ι
    i : ι
    this : Decidable (Disjoint (f i) (t.sup f))
    ⊢ Decidable (Not (Membership.mem t i) → Disjoint (f i) (t.sup f))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


theorem SupIndep.subset (ht : t.SupIndep f) (h : s ⊆ t) : s.SupIndep f := fun _ hu _ hi =>
  ht (hu.trans h) (h hi)


@[simp]
theorem supIndep_empty (f : ι → α) : (∅ : Finset ι).SupIndep f := fun _ _ a ha =>
  (not_mem_empty a ha).elim


theorem supIndep_singleton (i : ι) (f : ι → α) : ({i} : Finset ι).SupIndep f :=
  fun s hs j hji hj => by
    /-
      α : Type u_1
      ι : Type u_3
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      i : ι
      f : ι → α
      s : Finset ι
      hs : HasSubset.Subset s (Singleton.singleton i)
      j : ι
      hji : Membership.mem (Singleton.singleton i) j
      hj : Not (Membership.mem s j)
      ⊢ Disjoint (f j) (s.sup f)
    -/
    rw [eq_empty_of_ssubset_singleton ⟨hs, fun h => hj (h hji)⟩, sup_empty]
    /-
      α : Type u_1
      ι : Type u_3
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      i : ι
      f : ι → α
      s : Finset ι
      hs : HasSubset.Subset s (Singleton.singleton i)
      j : ι
      hji : Membership.mem (Singleton.singleton i) j
      hj : Not (Membership.mem s j)
      ⊢ Disjoint (f j) Bot.bot
    -/
    exact disjoint_bot_right
    /-
      🎉 no goals
    -/


theorem SupIndep.pairwiseDisjoint (hs : s.SupIndep f) : (s : Set ι).PairwiseDisjoint f :=
  fun _ ha _ hb hab =>
    sup_singleton.subst <| hs (singleton_subset_iff.2 hb) ha <| not_mem_singleton.2 hab


theorem SupIndep.le_sup_iff (hs : s.SupIndep f) (hts : t ⊆ s) (hi : i ∈ s) (hf : ∀ i, f i ≠ ⊥) :
    f i ≤ t.sup f ↔ i ∈ t := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s t : Finset ι
    f : ι → α
    i : ι
    hs : s.SupIndep f
    hts : HasSubset.Subset t s
    hi : Membership.mem s i
    hf : ∀ (i : ι), Ne (f i) Bot.bot
    ⊢ Iff (LE.le (f i) (t.sup f)) (Membership.mem t i)
  -/
  refine ⟨fun h => ?_, le_sup⟩
  /-
    α : Type u_1
    ι : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s t : Finset ι
    f : ι → α
    i : ι
    hs : s.SupIndep f
    hts : HasSubset.Subset t s
    hi : Membership.mem s i
    hf : ∀ (i : ι), Ne (f i) Bot.bot
    h : LE.le (f i) (t.sup f)
    ⊢ Membership.mem t i
  -/
  by_contra hit
  /-
    α : Type u_1
    ι : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s t : Finset ι
    f : ι → α
    i : ι
    hs : s.SupIndep f
    hts : HasSubset.Subset t s
    hi : Membership.mem s i
    hf : ∀ (i : ι), Ne (f i) Bot.bot
    h : LE.le (f i) (t.sup f)
    hit : Not (Membership.mem t i)
    ⊢ False
  -/
  exact hf i (disjoint_self.1 <| (hs hts hi hit).mono_right h)
  /-
    🎉 no goals
  -/


/-- The RHS looks like the definition of `iSupIndep`. -/
theorem supIndep_iff_disjoint_erase [DecidableEq ι] :
    s.SupIndep f ↔ ∀ i ∈ s, Disjoint (f i) ((s.erase i).sup f) :=
  ⟨fun hs _ hi => hs (erase_subset _ _) hi (not_mem_erase _ _), fun hs _ ht i hi hit =>
    (hs i hi).mono_right (sup_mono fun _ hj => mem_erase.2 ⟨ne_of_mem_of_not_mem hj hit, ht hj⟩)⟩


theorem supIndep_antimono_fun {g : ι → α} (h : ∀ x ∈ s, f x ≤ g x) (h : s.SupIndep g) :
    s.SupIndep f := by
  classical
  induction s using Finset.induction_on with
  | empty => apply Finset.supIndep_empty
  | @insert i s his IH =>
  rename_i hle
  rw [Finset.supIndep_iff_disjoint_erase] at h ⊢
  intro j hj
  simp_all only [Finset.mem_insert, or_true, implies_true, true_implies, forall_eq_or_imp,
    Finset.erase_insert_eq_erase, not_false_eq_true, Finset.erase_eq_of_not_mem]
  obtain rfl | hj := hj
  · simp only [Finset.erase_insert_eq_erase]
    apply h.left.mono hle.left
    apply (Finset.sup_mono _).trans (Finset.sup_mono_fun hle.right)
    exact Finset.erase_subset _ _
  · apply (h.right j hj).mono (hle.right j hj) (Finset.sup_mono_fun _)
    intro k hk
    simp only [Finset.mem_erase, ne_eq, Finset.mem_insert] at hk
    obtain ⟨-, rfl | hk⟩ := hk
    · exact hle.left
    · exact hle.right k hk


theorem SupIndep.image [DecidableEq ι] {s : Finset ι'} {g : ι' → ι} (hs : s.SupIndep (f ∘ g)) :
    (s.image g).SupIndep f := by
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    f : ι → α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → ι
    hs : s.SupIndep (Function.comp f g)
    ⊢ (Finset.image g s).SupIndep f
  -/
  intro t ht i hi hit
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    f : ι → α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → ι
    hs : s.SupIndep (Function.comp f g)
    t : Finset ι
    ht : HasSubset.Subset t (Finset.image g s)
    i : ι
    hi : Membership.mem (Finset.image g s) i
    hit : Not (Membership.mem t i)
    ⊢ Disjoint (f i) (t.sup f)
  -/
  rw [mem_image] at hi
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    f : ι → α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → ι
    hs : s.SupIndep (Function.comp f g)
    t : Finset ι
    ht : HasSubset.Subset t (Finset.image g s)
    i : ι
    hi : Exists fun a => And (Membership.mem s a) (Eq (g a) i)
    hit : Not (Membership.mem t i)
    ⊢ Disjoint (f i) (t.sup f)
  -/
  obtain ⟨i, hi, rfl⟩ := hi
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    f : ι → α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → ι
    hs : s.SupIndep (Function.comp f g)
    t : Finset ι
    ht : HasSubset.Subset t (Finset.image g s)
    i : ι'
    hi : Membership.mem s i
    hit : Not (Membership.mem t (g i))
    ⊢ Disjoint (f (g i)) (t.sup f)
  -/
  haveI : DecidableEq ι' := Classical.decEq _
  suffices hts : t ⊆ (s.erase i).image g by
    refine (supIndep_iff_disjoint_erase.1 hs i hi).mono_right ((sup_mono hts).trans ?_)
    rw [sup_image]
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    f : ι → α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → ι
    hs : s.SupIndep (Function.comp f g)
    t : Finset ι
    ht : HasSubset.Subset t (Finset.image g s)
    i : ι'
    hi : Membership.mem s i
    hit : Not (Membership.mem t (g i))
    this : DecidableEq ι'
    ⊢ HasSubset.Subset t (Finset.image g (s.erase i))
  -/
  rintro j hjt
  /-
    case intro.intro
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    f : ι → α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → ι
    hs : s.SupIndep (Function.comp f g)
    t : Finset ι
    ht : HasSubset.Subset t (Finset.image g s)
    i : ι'
    hi : Membership.mem s i
    hit : Not (Membership.mem t (g i))
    this : DecidableEq ι'
    j : ι
    hjt : Membership.mem t j
    ⊢ Membership.mem (Finset.image g (s.erase i)) j
  -/
  obtain ⟨j, hj, rfl⟩ := mem_image.1 (ht hjt)
  /-
    case intro.intro.intro.intro
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : Lattice α
    inst✝¹ : OrderBot α
    f : ι → α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → ι
    hs : s.SupIndep (Function.comp f g)
    t : Finset ι
    ht : HasSubset.Subset t (Finset.image g s)
    i : ι'
    hi : Membership.mem s i
    hit : Not (Membership.mem t (g i))
    this : DecidableEq ι'
    j : ι'
    hj : Membership.mem s j
    hjt : Membership.mem t (g j)
    ⊢ Membership.mem (Finset.image g (s.erase i)) (g j)
  -/
  exact mem_image_of_mem _ (mem_erase.2 ⟨ne_of_apply_ne g (ne_of_mem_of_not_mem hjt hit), hj⟩)
  /-
    🎉 no goals
  -/


theorem supIndep_map {s : Finset ι'} {g : ι' ↪ ι} : (s.map g).SupIndep f ↔ s.SupIndep (f ∘ g) := by
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    f : ι → α
    s : Finset ι'
    g : Function.Embedding ι' ι
    ⊢ Iff ((Finset.map g s).SupIndep f) (s.SupIndep (Function.comp f ⇑g))
  -/
  refine ⟨fun hs t ht i hi hit => ?_, fun hs => ?_⟩
    /-
      case refine_1
      α : Type u_1
      ι : Type u_3
      ι' : Type u_4
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      f : ι → α
      s : Finset ι'
      g : Function.Embedding ι' ι
      hs : (Finset.map g s).SupIndep f
      t : Finset ι'
      ht : HasSubset.Subset t s
      i : ι'
      hi : Membership.mem s i
      hit : Not (Membership.mem t i)
      ⊢ Disjoint (Function.comp f (⇑g) i) (t.sup (Function.comp f ⇑g))
    -/
  · rw [← sup_map]
    /-
      case refine_1
      α : Type u_1
      ι : Type u_3
      ι' : Type u_4
      inst✝¹ : Lattice α
      inst✝ : OrderBot α
      f : ι → α
      s : Finset ι'
      g : Function.Embedding ι' ι
      hs : (Finset.map g s).SupIndep f
      t : Finset ι'
      ht : HasSubset.Subset t s
      i : ι'
      hi : Membership.mem s i
      hit : Not (Membership.mem t i)
      ⊢ Disjoint (Function.comp f (⇑g) i) ((Finset.map g t).sup f)
    -/
    exact hs (map_subset_map.2 ht) ((mem_map' _).2 hi) (by rwa [mem_map'])
    /-
      🎉 no goals
    -/
  · classical
    rw [map_eq_image]
    exact hs.image


@[simp]
theorem supIndep_pair [DecidableEq ι] {i j : ι} (hij : i ≠ j) :
    ({i, j} : Finset ι).SupIndep f ↔ Disjoint (f i) (f j) :=
                                   /-
                                     α : Type u_1
                                     ι : Type u_3
                                     inst✝² : Lattice α
                                     inst✝¹ : OrderBot α
                                     f : ι → α
                                     inst✝ : DecidableEq ι
                                     i j : ι
                                     hij : Ne i j
                                     h : (Insert.insert i (Singleton.singleton j)).SupIndep f
                                     ⊢ Membership.mem (↑(Insert.insert i (Singleton.singleton j))) i
                                   -/
                                   /-
                                     🎉 no goals
                                   -/
  ⟨fun h => h.pairwiseDisjoint (by simp) (by simp) hij,
                                             /-
                                               🎉 no goals
                                             -/
   fun h => by
    /-
      α : Type u_1
      ι : Type u_3
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      f : ι → α
      inst✝ : DecidableEq ι
      i j : ι
      hij : Ne i j
      h : Disjoint (f i) (f j)
      ⊢ (Insert.insert i (Singleton.singleton j)).SupIndep f
    -/
    rw [supIndep_iff_disjoint_erase]
    /-
      α : Type u_1
      ι : Type u_3
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      f : ι → α
      inst✝ : DecidableEq ι
      i j : ι
      hij : Ne i j
      h : Disjoint (f i) (f j)
      ⊢ ∀ (i_1 : ι), Membership.mem (Insert.insert i (Singleton.singleton j)) i_1 →  …
    -/
    intro k hk
    /-
      α : Type u_1
      ι : Type u_3
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      f : ι → α
      inst✝ : DecidableEq ι
      i j : ι
      hij : Ne i j
      h : Disjoint (f i) (f j)
      k : ι
      hk : Membership.mem (Insert.insert i (Singleton.singleton j)) k
      ⊢ Disjoint (f k) (((Insert.insert i (Singleton.singleton j)).erase k).sup f)
    -/
    rw [Finset.mem_insert, Finset.mem_singleton] at hk
    /-
      α : Type u_1
      ι : Type u_3
      inst✝² : Lattice α
      inst✝¹ : OrderBot α
      f : ι → α
      inst✝ : DecidableEq ι
      i j : ι
      hij : Ne i j
      h : Disjoint (f i) (f j)
      k : ι
      hk : Or (Eq k i) (Eq k j)
      ⊢ Disjoint (f k) (((Insert.insert i (Singleton.singleton j)).erase k).sup f)
    -/
    obtain rfl | rfl := hk
      /-
        case inl
        α : Type u_1
        ι : Type u_3
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        f : ι → α
        inst✝ : DecidableEq ι
        j k : ι
        hij : Ne k j
        h : Disjoint (f k) (f j)
        ⊢ Disjoint (f k) (((Insert.insert k (Singleton.singleton j)).erase k).sup f)
      -/
    · convert h using 1
      /-
        case h.e'_5
        α : Type u_1
        ι : Type u_3
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        f : ι → α
        inst✝ : DecidableEq ι
        j k : ι
        hij : Ne k j
        h : Disjoint (f k) (f j)
        ⊢ Eq (((Insert.insert k (Singleton.singleton j)).erase k).sup f) (f j)
      -/
      rw [Finset.erase_insert, Finset.sup_singleton]
      /-
        case h.e'_5
        α : Type u_1
        ι : Type u_3
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        f : ι → α
        inst✝ : DecidableEq ι
        j k : ι
        hij : Ne k j
        h : Disjoint (f k) (f j)
        ⊢ Not (Membership.mem (Singleton.singleton j) k)
      -/
      simpa using hij
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_1
        ι : Type u_3
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        f : ι → α
        inst✝ : DecidableEq ι
        i k : ι
        hij : Ne i k
        h : Disjoint (f i) (f k)
        ⊢ Disjoint (f k) (((Insert.insert i (Singleton.singleton k)).erase k).sup f)
      -/
    · convert h.symm using 1
      have : ({i, k} : Finset ι).erase k = {i} := by
        ext
        rw [mem_erase, mem_insert, mem_singleton, mem_singleton, and_or_left, Ne,
          not_and_self_iff, or_false, and_iff_right_of_imp]
        rintro rfl
        exact hij
      /-
        case h.e'_5
        α : Type u_1
        ι : Type u_3
        inst✝² : Lattice α
        inst✝¹ : OrderBot α
        f : ι → α
        inst✝ : DecidableEq ι
        i k : ι
        hij : Ne i k
        h : Disjoint (f i) (f k)
        this : Eq ((Insert.insert i (Singleton.singleton k)).erase k) (Singleton.singl …
        ⊢ Eq (((Insert.insert i (Singleton.singleton k)).erase k).sup f) (f i)
      -/
      rw [this, Finset.sup_singleton]⟩
      /-
        🎉 no goals
      -/


theorem supIndep_univ_bool (f : Bool → α) :
    (Finset.univ : Finset Bool).SupIndep f ↔ Disjoint (f false) (f true) :=
                             /-
                               α : Type u_1
                               inst✝¹ : Lattice α
                               inst✝ : OrderBot α
                               f : Bool → α
                               ⊢ Ne Bool.true Bool.false
                             -/
  haveI : true ≠ false := by simp only [Ne, not_false_iff, reduceCtorEq]
                             /-
                               🎉 no goals
                             -/
  (supIndep_pair this).trans disjoint_comm


@[simp]
theorem supIndep_univ_fin_two (f : Fin 2 → α) :
    (Finset.univ : Finset (Fin 2)).SupIndep f ↔ Disjoint (f 0) (f 1) :=
                                /-
                                  α : Type u_1
                                  inst✝¹ : Lattice α
                                  inst✝ : OrderBot α
                                  f : Fin 2 → α
                                  ⊢ Ne 0 1
                                -/
  haveI : (0 : Fin 2) ≠ 1 := by simp
                                /-
                                  🎉 no goals
                                -/
  supIndep_pair this


theorem SupIndep.attach (hs : s.SupIndep f) : s.attach.SupIndep fun a => f a := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    hs : s.SupIndep f
    ⊢ s.attach.SupIndep fun a => f ↑a
  -/
  intro t _ i _ hi
  classical
    have : (fun (a : { x // x ∈ s }) => f ↑a) = f ∘ (fun a : { x // x ∈ s } => ↑a) := rfl
    rw [this, ← Finset.sup_image]
    refine hs (image_subset_iff.2 fun (j : { x // x ∈ s }) _ => j.2) i.2 fun hi' => hi ?_
    rw [mem_image] at hi'
    obtain ⟨j, hj, hji⟩ := hi'
    rwa [Subtype.ext hji] at hj


@[simp]
theorem supIndep_attach : (s.attach.SupIndep fun a => f a) ↔ s.SupIndep f := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝¹ : Lattice α
    inst✝ : OrderBot α
    s : Finset ι
    f : ι → α
    ⊢ Iff (s.attach.SupIndep fun a => f ↑a) (s.SupIndep f)
  -/
  refine ⟨fun h t ht i his hit => ?_, SupIndep.attach⟩
  classical
  convert h (filter_subset (fun (i : { x // x ∈ s }) => (i : ι) ∈ t) _) (mem_attach _ ⟨i, ‹_›⟩)
    fun hi => hit <| by simpa using hi using 1
  refine eq_of_forall_ge_iff ?_
  simp only [Finset.sup_le_iff, mem_filter, mem_attach, true_and, Function.comp_apply,
    Subtype.forall, Subtype.coe_mk]
  exact fun a => forall_congr' fun j => ⟨fun h _ => h, fun h hj => h (ht hj) hj⟩


theorem supIndep_iff_pairwiseDisjoint : s.SupIndep f ↔ (s : Set ι).PairwiseDisjoint f :=
  ⟨SupIndep.pairwiseDisjoint, fun hs _ ht _ hi hit =>
    Finset.disjoint_sup_right.2 fun _ hj => hs hi (ht hj) (ne_of_mem_of_not_mem hj hit).symm⟩


alias ⟨sup_indep.pairwise_disjoint, _root_.Set.PairwiseDisjoint.supIndep⟩ :=
  supIndep_iff_pairwiseDisjoint


/-- Bind operation for `SupIndep`. -/
theorem SupIndep.sup [DecidableEq ι] {s : Finset ι'} {g : ι' → Finset ι} {f : ι → α}
    (hs : s.SupIndep fun i => (g i).sup f) (hg : ∀ i' ∈ s, (g i').SupIndep f) :
    (s.sup g).SupIndep f := by
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.SupIndep fun i => (g i).sup f
    hg : ∀ (i' : ι'), Membership.mem s i' → (g i').SupIndep f
    ⊢ (s.sup g).SupIndep f
  -/
  simp_rw [supIndep_iff_pairwiseDisjoint] at hs hg ⊢
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → Finset ι
    f : ι → α
    hs : (↑s).PairwiseDisjoint fun i => (g i).sup f
    hg : ∀ (i' : ι'), Membership.mem s i' → (↑(g i')).PairwiseDisjoint f
    ⊢ (↑(s.sup g)).PairwiseDisjoint f
  -/
  rw [sup_eq_biUnion, coe_biUnion]
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → Finset ι
    f : ι → α
    hs : (↑s).PairwiseDisjoint fun i => (g i).sup f
    hg : ∀ (i' : ι'), Membership.mem s i' → (↑(g i')).PairwiseDisjoint f
    ⊢ (Set.iUnion fun x => Set.iUnion fun h => ↑(g x)).PairwiseDisjoint f
  -/
  exact hs.biUnion_finset hg
  /-
    🎉 no goals
  -/


/-- Bind operation for `SupIndep`. -/
theorem SupIndep.biUnion [DecidableEq ι] {s : Finset ι'} {g : ι' → Finset ι} {f : ι → α}
    (hs : s.SupIndep fun i => (g i).sup f) (hg : ∀ i' ∈ s, (g i').SupIndep f) :
    (s.biUnion g).SupIndep f := by
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.SupIndep fun i => (g i).sup f
    hg : ∀ (i' : ι'), Membership.mem s i' → (g i').SupIndep f
    ⊢ (s.biUnion g).SupIndep f
  -/
  rw [← sup_eq_biUnion]
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableEq ι
    s : Finset ι'
    g : ι' → Finset ι
    f : ι → α
    hs : s.SupIndep fun i => (g i).sup f
    hg : ∀ (i' : ι'), Membership.mem s i' → (g i').SupIndep f
    ⊢ (s.sup g).SupIndep f
  -/
  exact hs.sup hg
  /-
    🎉 no goals
  -/


/-- Bind operation for `SupIndep`. -/
theorem SupIndep.sigma {β : ι → Type*} {s : Finset ι} {g : ∀ i, Finset (β i)} {f : Sigma β → α}
    (hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩)
    (hg : ∀ i ∈ s, (g i).SupIndep fun b => f ⟨i, b⟩) : (s.sigma g).SupIndep f := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    β : ι → Type u_5
    s : Finset ι
    g : (i : ι) → Finset (β i)
    f : Sigma β → α
    hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
    hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
    ⊢ (s.sigma g).SupIndep f
  -/
  rintro t ht ⟨i, b⟩ hi hit
  /-
    case mk
    α : Type u_1
    ι : Type u_3
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    β : ι → Type u_5
    s : Finset ι
    g : (i : ι) → Finset (β i)
    f : Sigma β → α
    hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
    hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
    t : Finset (Sigma fun i => β i)
    ht : HasSubset.Subset t (s.sigma g)
    i : ι
    b : β i
    hi : Membership.mem (s.sigma g) ⟨i, b⟩
    hit : Not (Membership.mem t ⟨i, b⟩)
    ⊢ Disjoint (f ⟨i, b⟩) (t.sup f)
  -/
  rw [Finset.disjoint_sup_right]
  /-
    case mk
    α : Type u_1
    ι : Type u_3
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    β : ι → Type u_5
    s : Finset ι
    g : (i : ι) → Finset (β i)
    f : Sigma β → α
    hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
    hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
    t : Finset (Sigma fun i => β i)
    ht : HasSubset.Subset t (s.sigma g)
    i : ι
    b : β i
    hi : Membership.mem (s.sigma g) ⟨i, b⟩
    hit : Not (Membership.mem t ⟨i, b⟩)
    ⊢ ∀ ⦃i_1 : Sigma fun i => β i⦄, Membership.mem t i_1 → Disjoint (f ⟨i, b⟩) (f  …
  -/
  rintro ⟨j, c⟩ hj
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    β : ι → Type u_5
    s : Finset ι
    g : (i : ι) → Finset (β i)
    f : Sigma β → α
    hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
    hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
    t : Finset (Sigma fun i => β i)
    ht : HasSubset.Subset t (s.sigma g)
    i : ι
    b : β i
    hi : Membership.mem (s.sigma g) ⟨i, b⟩
    hit : Not (Membership.mem t ⟨i, b⟩)
    j : ι
    c : β j
    hj : Membership.mem t ⟨j, c⟩
    ⊢ Disjoint (f ⟨i, b⟩) (f ⟨j, c⟩)
  -/
  have hbc := (ne_of_mem_of_not_mem hj hit).symm
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    β : ι → Type u_5
    s : Finset ι
    g : (i : ι) → Finset (β i)
    f : Sigma β → α
    hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
    hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
    t : Finset (Sigma fun i => β i)
    ht : HasSubset.Subset t (s.sigma g)
    i : ι
    b : β i
    hi : Membership.mem (s.sigma g) ⟨i, b⟩
    hit : Not (Membership.mem t ⟨i, b⟩)
    j : ι
    c : β j
    hj : Membership.mem t ⟨j, c⟩
    hbc : Ne ⟨i, b⟩ ⟨j, c⟩
    ⊢ Disjoint (f ⟨i, b⟩) (f ⟨j, c⟩)
  -/
  replace hj := ht hj
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    β : ι → Type u_5
    s : Finset ι
    g : (i : ι) → Finset (β i)
    f : Sigma β → α
    hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
    hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
    t : Finset (Sigma fun i => β i)
    ht : HasSubset.Subset t (s.sigma g)
    i : ι
    b : β i
    hi : Membership.mem (s.sigma g) ⟨i, b⟩
    hit : Not (Membership.mem t ⟨i, b⟩)
    j : ι
    c : β j
    hbc : Ne ⟨i, b⟩ ⟨j, c⟩
    hj : Membership.mem (s.sigma g) ⟨j, c⟩
    ⊢ Disjoint (f ⟨i, b⟩) (f ⟨j, c⟩)
  -/
  rw [mem_sigma] at hi hj
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    β : ι → Type u_5
    s : Finset ι
    g : (i : ι) → Finset (β i)
    f : Sigma β → α
    hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
    hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
    t : Finset (Sigma fun i => β i)
    ht : HasSubset.Subset t (s.sigma g)
    i : ι
    b : β i
    hi : And (Membership.mem s ⟨i, b⟩.fst) (Membership.mem (g ⟨i, b⟩.fst) ⟨i, b⟩.s …
    hit : Not (Membership.mem t ⟨i, b⟩)
    j : ι
    c : β j
    hbc : Ne ⟨i, b⟩ ⟨j, c⟩
    hj : And (Membership.mem s ⟨j, c⟩.fst) (Membership.mem (g ⟨j, c⟩.fst) ⟨j, c⟩.s …
    ⊢ Disjoint (f ⟨i, b⟩) (f ⟨j, c⟩)
  -/
  obtain rfl | hij := eq_or_ne i j
    /-
      case mk.mk.inl
      α : Type u_1
      ι : Type u_3
      inst✝¹ : DistribLattice α
      inst✝ : OrderBot α
      β : ι → Type u_5
      s : Finset ι
      g : (i : ι) → Finset (β i)
      f : Sigma β → α
      hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
      hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
      t : Finset (Sigma fun i => β i)
      ht : HasSubset.Subset t (s.sigma g)
      i : ι
      b : β i
      hi : And (Membership.mem s ⟨i, b⟩.fst) (Membership.mem (g ⟨i, b⟩.fst) ⟨i, b⟩.s …
      hit : Not (Membership.mem t ⟨i, b⟩)
      c : β i
      hbc : Ne ⟨i, b⟩ ⟨i, c⟩
      hj : And (Membership.mem s ⟨i, c⟩.fst) (Membership.mem (g ⟨i, c⟩.fst) ⟨i, c⟩.s …
      ⊢ Disjoint (f ⟨i, b⟩) (f ⟨i, c⟩)
    -/
  · exact (hg _ hj.1).pairwiseDisjoint hi.2 hj.2 (sigma_mk_injective.ne_iff.1 hbc)
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.inr
      α : Type u_1
      ι : Type u_3
      inst✝¹ : DistribLattice α
      inst✝ : OrderBot α
      β : ι → Type u_5
      s : Finset ι
      g : (i : ι) → Finset (β i)
      f : Sigma β → α
      hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
      hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
      t : Finset (Sigma fun i => β i)
      ht : HasSubset.Subset t (s.sigma g)
      i : ι
      b : β i
      hi : And (Membership.mem s ⟨i, b⟩.fst) (Membership.mem (g ⟨i, b⟩.fst) ⟨i, b⟩.s …
      hit : Not (Membership.mem t ⟨i, b⟩)
      j : ι
      c : β j
      hbc : Ne ⟨i, b⟩ ⟨j, c⟩
      hj : And (Membership.mem s ⟨j, c⟩.fst) (Membership.mem (g ⟨j, c⟩.fst) ⟨j, c⟩.s …
      hij : Ne i j
      ⊢ Disjoint (f ⟨i, b⟩) (f ⟨j, c⟩)
    -/
  · refine (hs.pairwiseDisjoint hi.1 hj.1 hij).mono ?_ ?_
      /-
        case mk.mk.inr.refine_1
        α : Type u_1
        ι : Type u_3
        inst✝¹ : DistribLattice α
        inst✝ : OrderBot α
        β : ι → Type u_5
        s : Finset ι
        g : (i : ι) → Finset (β i)
        f : Sigma β → α
        hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
        hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
        t : Finset (Sigma fun i => β i)
        ht : HasSubset.Subset t (s.sigma g)
        i : ι
        b : β i
        hi : And (Membership.mem s ⟨i, b⟩.fst) (Membership.mem (g ⟨i, b⟩.fst) ⟨i, b⟩.s …
        hit : Not (Membership.mem t ⟨i, b⟩)
        j : ι
        c : β j
        hbc : Ne ⟨i, b⟩ ⟨j, c⟩
        hj : And (Membership.mem s ⟨j, c⟩.fst) (Membership.mem (g ⟨j, c⟩.fst) ⟨j, c⟩.s …
        hij : Ne i j
        ⊢ LE.le (f ⟨i, b⟩) ((fun i => (g i).sup fun b => f ⟨i, b⟩) ⟨i, b⟩.fst)
      -/
    · convert le_sup (α := α) hi.2; simp
                                    /-
                                      🎉 no goals
                                    -/
      /-
        case mk.mk.inr.refine_2
        α : Type u_1
        ι : Type u_3
        inst✝¹ : DistribLattice α
        inst✝ : OrderBot α
        β : ι → Type u_5
        s : Finset ι
        g : (i : ι) → Finset (β i)
        f : Sigma β → α
        hs : s.SupIndep fun i => (g i).sup fun b => f ⟨i, b⟩
        hg : ∀ (i : ι), Membership.mem s i → (g i).SupIndep fun b => f ⟨i, b⟩
        t : Finset (Sigma fun i => β i)
        ht : HasSubset.Subset t (s.sigma g)
        i : ι
        b : β i
        hi : And (Membership.mem s ⟨i, b⟩.fst) (Membership.mem (g ⟨i, b⟩.fst) ⟨i, b⟩.s …
        hit : Not (Membership.mem t ⟨i, b⟩)
        j : ι
        c : β j
        hbc : Ne ⟨i, b⟩ ⟨j, c⟩
        hj : And (Membership.mem s ⟨j, c⟩.fst) (Membership.mem (g ⟨j, c⟩.fst) ⟨j, c⟩.s …
        hij : Ne i j
        ⊢ LE.le (f ⟨j, c⟩) ((fun i => (g i).sup fun b => f ⟨i, b⟩) ⟨j, c⟩.fst)
      -/
    · convert le_sup (α := α) hj.2; simp
                                    /-
                                      🎉 no goals
                                    -/


theorem SupIndep.product {s : Finset ι} {t : Finset ι'} {f : ι × ι' → α}
    (hs : s.SupIndep fun i => t.sup fun i' => f (i, i'))
    (ht : t.SupIndep fun i' => s.sup fun i => f (i, i')) : (s ×ˢ t).SupIndep f := by
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
    ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
    ⊢ (SProd.sprod s t).SupIndep f
  -/
  rintro u hu ⟨i, i'⟩ hi hiu
  /-
    case mk
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
    ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
    u : Finset (Prod ι ι')
    hu : HasSubset.Subset u (SProd.sprod s t)
    i : ι
    i' : ι'
    hi : Membership.mem (SProd.sprod s t) { fst := i, snd := i' }
    hiu : Not (Membership.mem u { fst := i, snd := i' })
    ⊢ Disjoint (f { fst := i, snd := i' }) (u.sup f)
  -/
  rw [Finset.disjoint_sup_right]
  /-
    case mk
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
    ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
    u : Finset (Prod ι ι')
    hu : HasSubset.Subset u (SProd.sprod s t)
    i : ι
    i' : ι'
    hi : Membership.mem (SProd.sprod s t) { fst := i, snd := i' }
    hiu : Not (Membership.mem u { fst := i, snd := i' })
    ⊢ ∀ ⦃i_1 : Prod ι ι'⦄, Membership.mem u i_1 → Disjoint (f { fst := i, snd := i …
  -/
  rintro ⟨j, j'⟩ hj
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
    ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
    u : Finset (Prod ι ι')
    hu : HasSubset.Subset u (SProd.sprod s t)
    i : ι
    i' : ι'
    hi : Membership.mem (SProd.sprod s t) { fst := i, snd := i' }
    hiu : Not (Membership.mem u { fst := i, snd := i' })
    j : ι
    j' : ι'
    hj : Membership.mem u { fst := j, snd := j' }
    ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := j, snd := j' })
  -/
  have hij := (ne_of_mem_of_not_mem hj hiu).symm
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
    ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
    u : Finset (Prod ι ι')
    hu : HasSubset.Subset u (SProd.sprod s t)
    i : ι
    i' : ι'
    hi : Membership.mem (SProd.sprod s t) { fst := i, snd := i' }
    hiu : Not (Membership.mem u { fst := i, snd := i' })
    j : ι
    j' : ι'
    hj : Membership.mem u { fst := j, snd := j' }
    hij : Ne { fst := i, snd := i' } { fst := j, snd := j' }
    ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := j, snd := j' })
  -/
  replace hj := hu hj
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
    ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
    u : Finset (Prod ι ι')
    hu : HasSubset.Subset u (SProd.sprod s t)
    i : ι
    i' : ι'
    hi : Membership.mem (SProd.sprod s t) { fst := i, snd := i' }
    hiu : Not (Membership.mem u { fst := i, snd := i' })
    j : ι
    j' : ι'
    hij : Ne { fst := i, snd := i' } { fst := j, snd := j' }
    hj : Membership.mem (SProd.sprod s t) { fst := j, snd := j' }
    ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := j, snd := j' })
  -/
  rw [mem_product] at hi hj
  /-
    case mk.mk
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
    ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
    u : Finset (Prod ι ι')
    hu : HasSubset.Subset u (SProd.sprod s t)
    i : ι
    i' : ι'
    hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
    hiu : Not (Membership.mem u { fst := i, snd := i' })
    j : ι
    j' : ι'
    hij : Ne { fst := i, snd := i' } { fst := j, snd := j' }
    hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
    ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := j, snd := j' })
  -/
  obtain rfl | hij := eq_or_ne i j
    /-
      case mk.mk.inl
      α : Type u_1
      ι : Type u_3
      ι' : Type u_4
      inst✝¹ : DistribLattice α
      inst✝ : OrderBot α
      s : Finset ι
      t : Finset ι'
      f : Prod ι ι' → α
      hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
      ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
      u : Finset (Prod ι ι')
      hu : HasSubset.Subset u (SProd.sprod s t)
      i : ι
      i' : ι'
      hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
      hiu : Not (Membership.mem u { fst := i, snd := i' })
      j' : ι'
      hij : Ne { fst := i, snd := i' } { fst := i, snd := j' }
      hj : And (Membership.mem s { fst := i, snd := j' }.1) (Membership.mem t { fst  …
      ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := i, snd := j' })
    -/
  · refine (ht.pairwiseDisjoint hi.2 hj.2 <| (Prod.mk.inj_left _).ne_iff.1 hij).mono ?_ ?_
      /-
        case mk.mk.inl.refine_1
        α : Type u_1
        ι : Type u_3
        ι' : Type u_4
        inst✝¹ : DistribLattice α
        inst✝ : OrderBot α
        s : Finset ι
        t : Finset ι'
        f : Prod ι ι' → α
        hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
        ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
        u : Finset (Prod ι ι')
        hu : HasSubset.Subset u (SProd.sprod s t)
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        hiu : Not (Membership.mem u { fst := i, snd := i' })
        j' : ι'
        hij : Ne { fst := i, snd := i' } { fst := i, snd := j' }
        hj : And (Membership.mem s { fst := i, snd := j' }.1) (Membership.mem t { fst  …
        ⊢ LE.le (f { fst := i, snd := i' }) ((fun i' => s.sup fun i => f { fst := i, s …
      -/
    · convert le_sup (α := α) hi.1; simp
                                    /-
                                      🎉 no goals
                                    -/
      /-
        case mk.mk.inl.refine_2
        α : Type u_1
        ι : Type u_3
        ι' : Type u_4
        inst✝¹ : DistribLattice α
        inst✝ : OrderBot α
        s : Finset ι
        t : Finset ι'
        f : Prod ι ι' → α
        hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
        ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
        u : Finset (Prod ι ι')
        hu : HasSubset.Subset u (SProd.sprod s t)
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        hiu : Not (Membership.mem u { fst := i, snd := i' })
        j' : ι'
        hij : Ne { fst := i, snd := i' } { fst := i, snd := j' }
        hj : And (Membership.mem s { fst := i, snd := j' }.1) (Membership.mem t { fst  …
        ⊢ LE.le (f { fst := i, snd := j' }) ((fun i' => s.sup fun i => f { fst := i, s …
      -/
    · convert le_sup (α := α) hj.1; simp
                                    /-
                                      🎉 no goals
                                    -/
    /-
      case mk.mk.inr
      α : Type u_1
      ι : Type u_3
      ι' : Type u_4
      inst✝¹ : DistribLattice α
      inst✝ : OrderBot α
      s : Finset ι
      t : Finset ι'
      f : Prod ι ι' → α
      hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
      ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
      u : Finset (Prod ι ι')
      hu : HasSubset.Subset u (SProd.sprod s t)
      i : ι
      i' : ι'
      hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
      hiu : Not (Membership.mem u { fst := i, snd := i' })
      j : ι
      j' : ι'
      hij✝ : Ne { fst := i, snd := i' } { fst := j, snd := j' }
      hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
      hij : Ne i j
      ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := j, snd := j' })
    -/
  · refine (hs.pairwiseDisjoint hi.1 hj.1 hij).mono ?_ ?_
      /-
        case mk.mk.inr.refine_1
        α : Type u_1
        ι : Type u_3
        ι' : Type u_4
        inst✝¹ : DistribLattice α
        inst✝ : OrderBot α
        s : Finset ι
        t : Finset ι'
        f : Prod ι ι' → α
        hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
        ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
        u : Finset (Prod ι ι')
        hu : HasSubset.Subset u (SProd.sprod s t)
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        hiu : Not (Membership.mem u { fst := i, snd := i' })
        j : ι
        j' : ι'
        hij✝ : Ne { fst := i, snd := i' } { fst := j, snd := j' }
        hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
        hij : Ne i j
        ⊢ LE.le (f { fst := i, snd := i' }) ((fun i => t.sup fun i' => f { fst := i, s …
      -/
    · convert le_sup (α := α) hi.2; simp
                                    /-
                                      🎉 no goals
                                    -/
      /-
        case mk.mk.inr.refine_2
        α : Type u_1
        ι : Type u_3
        ι' : Type u_4
        inst✝¹ : DistribLattice α
        inst✝ : OrderBot α
        s : Finset ι
        t : Finset ι'
        f : Prod ι ι' → α
        hs : s.SupIndep fun i => t.sup fun i' => f { fst := i, snd := i' }
        ht : t.SupIndep fun i' => s.sup fun i => f { fst := i, snd := i' }
        u : Finset (Prod ι ι')
        hu : HasSubset.Subset u (SProd.sprod s t)
        i : ι
        i' : ι'
        hi : And (Membership.mem s { fst := i, snd := i' }.1) (Membership.mem t { fst  …
        hiu : Not (Membership.mem u { fst := i, snd := i' })
        j : ι
        j' : ι'
        hij✝ : Ne { fst := i, snd := i' } { fst := j, snd := j' }
        hj : And (Membership.mem s { fst := j, snd := j' }.1) (Membership.mem t { fst  …
        hij : Ne i j
        ⊢ LE.le (f { fst := j, snd := j' }) ((fun i => t.sup fun i' => f { fst := i, s …
      -/
    · convert le_sup (α := α) hj.2; simp
                                    /-
                                      🎉 no goals
                                    -/


theorem supIndep_product_iff {s : Finset ι} {t : Finset ι'} {f : ι × ι' → α} :
    (s.product t).SupIndep f ↔ (s.SupIndep fun i => t.sup fun i' => f (i, i'))
      ∧ t.SupIndep fun i' => s.sup fun i => f (i, i') := by
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    ⊢ Iff ((s.product t).SupIndep f) (And (s.SupIndep fun i => t.sup fun i' => f { …
  -/
  refine ⟨?_, fun h => h.1.product h.2⟩
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    ⊢ (s.product t).SupIndep f → And (s.SupIndep fun i => t.sup fun i' => f { fst  …
  -/
  simp_rw [supIndep_iff_pairwiseDisjoint]
  /-
    α : Type u_1
    ι : Type u_3
    ι' : Type u_4
    inst✝¹ : DistribLattice α
    inst✝ : OrderBot α
    s : Finset ι
    t : Finset ι'
    f : Prod ι ι' → α
    ⊢ (↑(s.product t)).PairwiseDisjoint f → And ((↑s).PairwiseDisjoint fun i => t. …
  -/
  refine fun h => ⟨fun i hi j hj hij => ?_, fun i hi j hj hij => ?_⟩ <;>
      /-
        case refine_1
        α : Type u_1
        ι : Type u_3
        ι' : Type u_4
        inst✝¹ : DistribLattice α
        inst✝ : OrderBot α
        s : Finset ι
        t : Finset ι'
        f : Prod ι ι' → α
        h : (↑(s.product t)).PairwiseDisjoint f
        i : ι
        hi : Membership.mem (↑s) i
        j : ι
        hj : Membership.mem (↑s) j
        hij : Ne i j
        ⊢ Function.onFun Disjoint (fun i => t.sup fun i' => f { fst := i, snd := i' }) …
      -/
      simp_rw [Finset.disjoint_sup_left, Finset.disjoint_sup_right] <;>
    /-
      case refine_1
      α : Type u_1
      ι : Type u_3
      ι' : Type u_4
      inst✝¹ : DistribLattice α
      inst✝ : OrderBot α
      s : Finset ι
      t : Finset ι'
      f : Prod ι ι' → α
      h : (↑(s.product t)).PairwiseDisjoint f
      i : ι
      hi : Membership.mem (↑s) i
      j : ι
      hj : Membership.mem (↑s) j
      hij : Ne i j
      ⊢ ∀ ⦃i_1 : ι'⦄, Membership.mem t i_1 → ∀ ⦃i_2 : ι'⦄, Membership.mem t i_2 → Di …
    -/
    intro i' hi' j' hj'
    /-
      case refine_1
      α : Type u_1
      ι : Type u_3
      ι' : Type u_4
      inst✝¹ : DistribLattice α
      inst✝ : OrderBot α
      s : Finset ι
      t : Finset ι'
      f : Prod ι ι' → α
      h : (↑(s.product t)).PairwiseDisjoint f
      i : ι
      hi : Membership.mem (↑s) i
      j : ι
      hj : Membership.mem (↑s) j
      hij : Ne i j
      i' : ι'
      hi' : Membership.mem t i'
      j' : ι'
      hj' : Membership.mem t j'
      ⊢ Disjoint (f { fst := i, snd := i' }) (f { fst := j, snd := j' })
    -/
  · exact h (mk_mem_product hi hi') (mk_mem_product hj hj') (ne_of_apply_ne Prod.fst hij)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Type u_3
      ι' : Type u_4
      inst✝¹ : DistribLattice α
      inst✝ : OrderBot α
      s : Finset ι
      t : Finset ι'
      f : Prod ι ι' → α
      h : (↑(s.product t)).PairwiseDisjoint f
      i : ι'
      hi : Membership.mem (↑t) i
      j : ι'
      hj : Membership.mem (↑t) j
      hij : Ne i j
      i' : ι
      hi' : Membership.mem s i'
      j' : ι
      hj' : Membership.mem s j'
      ⊢ Disjoint (f { fst := i', snd := i }) (f { fst := j', snd := j })
    -/
  · exact h (mk_mem_product hi' hi) (mk_mem_product hj' hj) (ne_of_apply_ne Prod.snd hij)
    /-
      🎉 no goals
    -/


/-- An independent set of elements in a complete lattice is one in which every element is disjoint
  from the `Sup` of the rest. -/
def sSupIndep (s : Set α) : Prop :=
  ∀ ⦃a⦄, a ∈ s → Disjoint a (sSup (s \ {a}))


@[deprecated (since := "2024-11-24")] alias CompleteLattice.SetIndependent := sSupIndep


@[simp]
theorem sSupIndep_empty : sSupIndep (∅ : Set α) := fun x hx =>
  (Set.not_mem_empty x hx).elim


@[deprecated (since := "2024-11-24")] alias CompleteLattice.setIndependent_empty := sSupIndep_empty


include hs in
theorem sSupIndep.mono {t : Set α} (hst : t ⊆ s) : sSupIndep t := fun _ ha =>
  (hs (hst ha)).mono_right (sSup_le_sSup (diff_subset_diff_left hst))


@[deprecated (since := "2024-11-24")] alias CompleteLattice.SetIndependent.mono := sSupIndep.mono


include hs in
/-- If the elements of a set are independent, then any pair within that set is disjoint. -/
theorem sSupIndep.pairwiseDisjoint : s.PairwiseDisjoint id := fun _ hx y hy h =>
  disjoint_sSup_right (hs hx) ((mem_diff y).mpr ⟨hy, h.symm⟩)


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.SetIndependent.pairwiseDisjoint := sSupIndep.pairwiseDisjoint


theorem sSupIndep_singleton (a : α) : sSupIndep ({a} : Set α) := fun i hi ↦ by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    a i : α
    hi : Membership.mem (Singleton.singleton a) i
    ⊢ Disjoint i (SupSet.sSup (SDiff.sdiff (Singleton.singleton a) (Singleton.sing …
  -/
  simp_all
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.setIndependent_singleton := sSupIndep_singleton


theorem sSupIndep_pair {a b : α} (hab : a ≠ b) :
    sSupIndep ({a, b} : Set α) ↔ Disjoint a b := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    a b : α
    hab : Ne a b
    ⊢ Iff (sSupIndep (Insert.insert a (Singleton.singleton b))) (Disjoint a b)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : CompleteLattice α
      a b : α
      hab : Ne a b
      ⊢ sSupIndep (Insert.insert a (Singleton.singleton b)) → Disjoint a b
    -/
  · intro h
    /-
      case mp
      α : Type u_1
      inst✝ : CompleteLattice α
      a b : α
      hab : Ne a b
      h : sSupIndep (Insert.insert a (Singleton.singleton b))
      ⊢ Disjoint a b
    -/
    exact h.pairwiseDisjoint (mem_insert _ _) (mem_insert_of_mem _ (mem_singleton _)) hab
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : CompleteLattice α
      a b : α
      hab : Ne a b
      ⊢ Disjoint a b → sSupIndep (Insert.insert a (Singleton.singleton b))
    -/
  · rintro h c ((rfl : c = a) | (rfl : c = b))
      /-
        case mpr.inl
        α : Type u_1
        inst✝ : CompleteLattice α
        b c : α
        hab : Ne c b
        h : Disjoint c b
        ⊢ Disjoint c (SupSet.sSup (SDiff.sdiff (Insert.insert c (Singleton.singleton b …
      -/
    · convert h using 1
      /-
        case h.e'_5
        α : Type u_1
        inst✝ : CompleteLattice α
        b c : α
        hab : Ne c b
        h : Disjoint c b
        ⊢ Eq (SupSet.sSup (SDiff.sdiff (Insert.insert c (Singleton.singleton b)) (Sing …
      -/
      simp [hab, sSup_singleton]
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u_1
        inst✝ : CompleteLattice α
        a c : α
        hab : Ne a c
        h : Disjoint a c
        ⊢ Disjoint c (SupSet.sSup (SDiff.sdiff (Insert.insert a (Singleton.singleton c …
      -/
    · convert h.symm using 1
      /-
        case h.e'_5
        α : Type u_1
        inst✝ : CompleteLattice α
        a c : α
        hab : Ne a c
        h : Disjoint a c
        ⊢ Eq (SupSet.sSup (SDiff.sdiff (Insert.insert a (Singleton.singleton c)) (Sing …
      -/
      simp [hab, sSup_singleton]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.setIndependent_pair := sSupIndep_pair


include hs in
/-- If the elements of a set are independent, then any element is disjoint from the `sSup` of some
subset of the rest. -/
theorem sSupIndep.disjoint_sSup {x : α} {y : Set α} (hx : x ∈ s) (hy : y ⊆ s) (hxy : x ∉ y) :
    Disjoint x (sSup y) := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    hs : sSupIndep s
    x : α
    y : Set α
    hx : Membership.mem s x
    hy : HasSubset.Subset y s
    hxy : Not (Membership.mem y x)
    ⊢ Disjoint x (SupSet.sSup y)
  -/
  have := (hs.mono <| insert_subset_iff.mpr ⟨hx, hy⟩) (mem_insert x _)
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    hs : sSupIndep s
    x : α
    y : Set α
    hx : Membership.mem s x
    hy : HasSubset.Subset y s
    hxy : Not (Membership.mem y x)
    this : Disjoint x (SupSet.sSup (SDiff.sdiff (Insert.insert x y) (Singleton.sin …
    ⊢ Disjoint x (SupSet.sSup y)
  -/
  rw [insert_diff_of_mem _ (mem_singleton _), diff_singleton_eq_self hxy] at this
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    s : Set α
    hs : sSupIndep s
    x : α
    y : Set α
    hx : Membership.mem s x
    hy : HasSubset.Subset y s
    hxy : Not (Membership.mem y x)
    this : Disjoint x (SupSet.sSup y)
    ⊢ Disjoint x (SupSet.sSup y)
  -/
  exact this
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.SetIndependent.disjoint_sSup := sSupIndep.disjoint_sSup


/-- An independent indexed family of elements in a complete lattice is one in which every element
  is disjoint from the `iSup` of the rest.

  Example: an indexed family of non-zero elements in a
  vector space is linearly independent iff the indexed family of subspaces they generate is
  independent in this sense.

  Example: an indexed family of submodules of a module is independent in this sense if
  and only the natural map from the direct sum of the submodules to the module is injective. -/
def iSupIndep {ι : Sort*} {α : Type*} [CompleteLattice α] (t : ι → α) : Prop :=
  ∀ i : ι, Disjoint (t i) (⨆ (j) (_ : j ≠ i), t j)


@[deprecated (since := "2024-11-24")] alias CompleteLattice.Independent := iSupIndep


theorem sSupIndep_iff {α : Type*} [CompleteLattice α] (s : Set α) :
    sSupIndep s ↔ iSupIndep ((↑) : s → α) := by
  /-
    α : Type u_5
    inst✝ : CompleteLattice α
    s : Set α
    ⊢ Iff (sSupIndep s) (iSupIndep Subtype.val)
  -/
  simp_rw [iSupIndep, sSupIndep, SetCoe.forall, sSup_eq_iSup]
  /-
    α : Type u_5
    inst✝ : CompleteLattice α
    s : Set α
    ⊢ Iff (∀ ⦃a : α⦄, Membership.mem s a → Disjoint a (iSup fun a_2 => iSup fun h  …
  -/
  refine forall₂_congr fun a ha => ?_
  /-
    α : Type u_5
    inst✝ : CompleteLattice α
    s : Set α
    a : α
    ha : Membership.mem s a
    ⊢ Iff (Disjoint a (iSup fun a_1 => iSup fun h => a_1)) (Disjoint a (iSup fun j …
  -/
  simp [iSup_subtype, iSup_and]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.setIndependent_iff := sSupIndep_iff


theorem iSupIndep_def : iSupIndep t ↔ ∀ i, Disjoint (t i) (⨆ (j) (_ : j ≠ i), t j) :=
  Iff.rfl


@[deprecated (since := "2024-11-24")] alias CompleteLattice.independent_def := iSupIndep_def


theorem iSupIndep_def' : iSupIndep t ↔ ∀ i, Disjoint (t i) (sSup (t '' { j | j ≠ i })) := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ⊢ Iff (iSupIndep t) (∀ (i : ι), Disjoint (t i) (SupSet.sSup (Set.image t (setO …
  -/
  simp_rw [sSup_image]
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ⊢ Iff (iSupIndep t) (∀ (i : ι), Disjoint (t i) (iSup fun a => iSup fun h => t  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.independent_def' := iSupIndep_def'


theorem iSupIndep_def'' :
    iSupIndep t ↔ ∀ i, Disjoint (t i) (sSup { a | ∃ j ≠ i, t j = a }) := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ⊢ Iff (iSupIndep t) (∀ (i : ι), Disjoint (t i) (SupSet.sSup (setOf fun a => Ex …
  -/
  rw [iSupIndep_def']
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ⊢ Iff (∀ (i : ι), Disjoint (t i) (SupSet.sSup (Set.image t (setOf fun j => Ne  …
  -/
  aesop
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.independent_def'' := iSupIndep_def''


@[simp]
theorem iSupIndep_empty (t : Empty → α) : iSupIndep t :=
  nofun


@[deprecated (since := "2024-11-24")] alias CompleteLattice.independent_empty := iSupIndep_empty


@[simp]
theorem iSupIndep_pempty (t : PEmpty → α) : iSupIndep t :=
  nofun


@[deprecated (since := "2024-11-24")] alias CompleteLattice.independent_pempty := iSupIndep_pempty


include ht in
/-- If the elements of a set are independent, then any pair within that set is disjoint. -/
theorem iSupIndep.pairwiseDisjoint : Pairwise (Disjoint on t) := fun x y h =>
  disjoint_sSup_right (ht x) ⟨y, iSup_pos h.symm⟩


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.pairwiseDisjoint := iSupIndep.pairwiseDisjoint


theorem iSupIndep.mono {s t : ι → α} (hs : iSupIndep s) (hst : t ≤ s) : iSupIndep t :=
  fun i => (hs i).mono (hst i) <| iSup₂_mono fun j _ => hst j


@[deprecated (since := "2024-11-24")] alias CompleteLattice.Independent.mono := iSupIndep.mono


/-- Composing an independent indexed family with an injective function on the index results in
another indepedendent indexed family. -/
theorem iSupIndep.comp {ι ι' : Sort*} {t : ι → α} {f : ι' → ι} (ht : iSupIndep t)
    (hf : Injective f) : iSupIndep (t ∘ f) := fun i =>
  (ht (f i)).mono_right <| by
    /-
      α : Type u_1
      inst✝ : CompleteLattice α
      ι : Sort u_5
      ι' : Sort u_6
      t : ι → α
      f : ι' → ι
      ht : iSupIndep t
      hf : Function.Injective f
      i : ι'
      ⊢ LE.le (iSup fun j => iSup fun x => Function.comp t f j) (iSup fun j => iSup  …
    -/
    refine (iSup_mono fun i => ?_).trans (iSup_comp_le _ f)
    /-
      α : Type u_1
      inst✝ : CompleteLattice α
      ι : Sort u_5
      ι' : Sort u_6
      t : ι → α
      f : ι' → ι
      ht : iSupIndep t
      hf : Function.Injective f
      i✝ i : ι'
      ⊢ LE.le (iSup fun x => Function.comp t f i) (iSup fun x => t (f i))
    -/
    exact iSup_const_mono hf.ne
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.Independent.comp := iSupIndep.comp


theorem iSupIndep.comp' {ι ι' : Sort*} {t : ι → α} {f : ι' → ι} (ht : iSupIndep <| t ∘ f)
    (hf : Surjective f) : iSupIndep t := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    ι' : Sort u_6
    t : ι → α
    f : ι' → ι
    ht : iSupIndep (Function.comp t f)
    hf : Function.Surjective f
    ⊢ iSupIndep t
  -/
  intro i
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    ι' : Sort u_6
    t : ι → α
    f : ι' → ι
    ht : iSupIndep (Function.comp t f)
    hf : Function.Surjective f
    i : ι
    ⊢ Disjoint (t i) (iSup fun j => iSup fun x => t j)
  -/
  obtain ⟨i', rfl⟩ := hf i
  /-
    case intro
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    ι' : Sort u_6
    t : ι → α
    f : ι' → ι
    ht : iSupIndep (Function.comp t f)
    hf : Function.Surjective f
    i' : ι'
    ⊢ Disjoint (t (f i')) (iSup fun j => iSup fun x => t j)
  -/
  rw [← hf.iSup_comp]
  /-
    case intro
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    ι' : Sort u_6
    t : ι → α
    f : ι' → ι
    ht : iSupIndep (Function.comp t f)
    hf : Function.Surjective f
    i' : ι'
    ⊢ Disjoint (t (f i')) (iSup fun x => iSup fun x_1 => t (f x))
  -/
  exact (ht i').mono_right (biSup_mono fun j' hij => mt (congr_arg f) hij)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.Independent.comp' := iSupIndep.comp'


theorem iSupIndep.sSupIndep_range (ht : iSupIndep t) : sSupIndep <| range t := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    ⊢ sSupIndep (Set.range t)
  -/
  rw [sSupIndep_iff]
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    ⊢ iSupIndep Subtype.val
  -/
  rw [← coe_comp_rangeFactorization t] at ht
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep (Function.comp Subtype.val (Set.rangeFactorization t))
    ⊢ iSupIndep Subtype.val
  -/
  exact ht.comp' surjective_onto_range
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.setIndependent_range := iSupIndep.sSupIndep_range


@[simp]
theorem iSupIndep_ne_bot :
    iSupIndep (fun i : {i // t i ≠ ⊥} ↦ t i) ↔ iSupIndep t := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ⊢ Iff (iSupIndep fun i => t ↑i) (iSupIndep t)
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.comp Subtype.val_injective⟩
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    h : iSupIndep fun i => t ↑i
    ⊢ iSupIndep t
  -/
  simp only [iSupIndep_def] at h ⊢
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    h : ∀ (i : Subtype fun i => Ne (t i) Bot.bot), Disjoint (t ↑i) (iSup fun j =>  …
    ⊢ ∀ (i : ι), Disjoint (t i) (iSup fun j => iSup fun x => t j)
  -/
  intro i
  cases eq_or_ne (t i) ⊥ with
  | inl hi => simp [hi]
  | inr hi => ?_
  /-
    case inr
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    h : ∀ (i : Subtype fun i => Ne (t i) Bot.bot), Disjoint (t ↑i) (iSup fun j =>  …
    i : ι
    hi : Ne (t i) Bot.bot
    ⊢ Disjoint (t i) (iSup fun j => iSup fun x => t j)
  -/
  convert h ⟨i, hi⟩
  /-
    case h.e'_5
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    h : ∀ (i : Subtype fun i => Ne (t i) Bot.bot), Disjoint (t ↑i) (iSup fun j =>  …
    i : ι
    hi : Ne (t i) Bot.bot
    ⊢ Eq (iSup fun j => iSup fun x => t j) (iSup fun j => iSup fun x => t ↑j)
  -/
  have : ∀ j, ⨆ (_ : t j = ⊥), t j = ⊥ := fun j ↦ by simp only [iSup_eq_bot, imp_self]
  /-
    case h.e'_5
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    h : ∀ (i : Subtype fun i => Ne (t i) Bot.bot), Disjoint (t ↑i) (iSup fun j =>  …
    i : ι
    hi : Ne (t i) Bot.bot
    this : ∀ (j : ι), Eq (iSup fun x => t j) Bot.bot
    ⊢ Eq (iSup fun j => iSup fun x => t j) (iSup fun j => iSup fun x => t ↑j)
  -/
  rw [iSup_split _ (fun j ↦ t j = ⊥), iSup_subtype]
  simp only [iSup_comm (ι' := _ ≠ i), this, ne_eq, sup_of_le_right, Subtype.mk.injEq, iSup_bot,
    bot_le]


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.independent_ne_bot_iff_independent := iSupIndep_ne_bot


theorem iSupIndep.injOn (ht : iSupIndep t) : InjOn t {i | t i ≠ ⊥} := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    ⊢ Set.InjOn t (setOf fun i => Ne (t i) Bot.bot)
  -/
  rintro i _ j (hj : t j ≠ ⊥) h
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    i : ι
    a✝ : Membership.mem (setOf fun i => Ne (t i) Bot.bot) i
    j : ι
    hj : Ne (t j) Bot.bot
    h : Eq (t i) (t j)
    ⊢ Eq i j
  -/
  by_contra! contra
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    i : ι
    a✝ : Membership.mem (setOf fun i => Ne (t i) Bot.bot) i
    j : ι
    hj : Ne (t j) Bot.bot
    h : Eq (t i) (t j)
    contra : Ne i j
    ⊢ False
  -/
  apply hj
  suffices t j ≤ ⨆ (k) (_ : k ≠ i), t k by
    replace ht := (ht i).mono_right this
    rwa [h, disjoint_self] at ht
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    i : ι
    a✝ : Membership.mem (setOf fun i => Ne (t i) Bot.bot) i
    j : ι
    hj : Ne (t j) Bot.bot
    h : Eq (t i) (t j)
    contra : Ne i j
    ⊢ LE.le (t j) (iSup fun k => iSup fun x => t k)
  -/
  replace contra : j ≠ i := Ne.symm contra
  -- Porting note: needs explicit `f`
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    i : ι
    a✝ : Membership.mem (setOf fun i => Ne (t i) Bot.bot) i
    j : ι
    hj : Ne (t j) Bot.bot
    h : Eq (t i) (t j)
    contra : Ne j i
    ⊢ LE.le (t j) (iSup fun k => iSup fun x => t k)
  -/
  exact le_iSup₂ (f := fun x _ ↦ t x) j contra
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.Independent.injOn := iSupIndep.injOn


theorem iSupIndep.injective (ht : iSupIndep t) (h_ne_bot : ∀ i, t i ≠ ⊥) : Injective t := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    h_ne_bot : ∀ (i : ι), Ne (t i) Bot.bot
    ⊢ Function.Injective t
  -/
  suffices univ = {i | t i ≠ ⊥} by rw [injective_iff_injOn_univ, this]; exact ht.injOn
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    ht : iSupIndep t
    h_ne_bot : ∀ (i : ι), Ne (t i) Bot.bot
    ⊢ Eq Set.univ (setOf fun i => Ne (t i) Bot.bot)
  -/
  aesop
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.injective := iSupIndep.injective


theorem iSupIndep_pair {i j : ι} (hij : i ≠ j) (huniv : ∀ k, k = i ∨ k = j) :
    iSupIndep t ↔ Disjoint (t i) (t j) := by
  /-
    α : Type u_1
    ι : Type u_3
    inst✝ : CompleteLattice α
    t : ι → α
    i j : ι
    hij : Ne i j
    huniv : ∀ (k : ι), Or (Eq k i) (Eq k j)
    ⊢ Iff (iSupIndep t) (Disjoint (t i) (t j))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      ι : Type u_3
      inst✝ : CompleteLattice α
      t : ι → α
      i j : ι
      hij : Ne i j
      huniv : ∀ (k : ι), Or (Eq k i) (Eq k j)
      ⊢ iSupIndep t → Disjoint (t i) (t j)
    -/
  · exact fun h => h.pairwiseDisjoint hij
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      ι : Type u_3
      inst✝ : CompleteLattice α
      t : ι → α
      i j : ι
      hij : Ne i j
      huniv : ∀ (k : ι), Or (Eq k i) (Eq k j)
      ⊢ Disjoint (t i) (t j) → iSupIndep t
    -/
  · rintro h k
    /-
      case mpr
      α : Type u_1
      ι : Type u_3
      inst✝ : CompleteLattice α
      t : ι → α
      i j : ι
      hij : Ne i j
      huniv : ∀ (k : ι), Or (Eq k i) (Eq k j)
      h : Disjoint (t i) (t j)
      k : ι
      ⊢ Disjoint (t k) (iSup fun j => iSup fun x => t j)
    -/
    obtain rfl | rfl := huniv k
      /-
        case mpr.inl
        α : Type u_1
        ι : Type u_3
        inst✝ : CompleteLattice α
        t : ι → α
        j k : ι
        hij : Ne k j
        huniv : ∀ (k_1 : ι), Or (Eq k_1 k) (Eq k_1 j)
        h : Disjoint (t k) (t j)
        ⊢ Disjoint (t k) (iSup fun j => iSup fun x => t j)
      -/
    · refine h.mono_right (iSup_le fun i => iSup_le fun hi => Eq.le ?_)
      /-
        case mpr.inl
        α : Type u_1
        ι : Type u_3
        inst✝ : CompleteLattice α
        t : ι → α
        j k : ι
        hij : Ne k j
        huniv : ∀ (k_1 : ι), Or (Eq k_1 k) (Eq k_1 j)
        h : Disjoint (t k) (t j)
        i : ι
        hi : Ne i k
        ⊢ Eq (t i) (t j)
      -/
      rw [(huniv i).resolve_left hi]
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        α : Type u_1
        ι : Type u_3
        inst✝ : CompleteLattice α
        t : ι → α
        i k : ι
        hij : Ne i k
        huniv : ∀ (k_1 : ι), Or (Eq k_1 i) (Eq k_1 k)
        h : Disjoint (t i) (t k)
        ⊢ Disjoint (t k) (iSup fun j => iSup fun x => t j)
      -/
    · refine h.symm.mono_right (iSup_le fun j => iSup_le fun hj => Eq.le ?_)
      /-
        case mpr.inr
        α : Type u_1
        ι : Type u_3
        inst✝ : CompleteLattice α
        t : ι → α
        i k : ι
        hij : Ne i k
        huniv : ∀ (k_1 : ι), Or (Eq k_1 i) (Eq k_1 k)
        h : Disjoint (t i) (t k)
        j : ι
        hj : Ne j k
        ⊢ Eq (t j) (t i)
      -/
      rw [(huniv j).resolve_right hj]
      /-
        🎉 no goals
      -/


@[deprecated (since := "2024-11-24")] alias CompleteLattice.independent_pair := iSupIndep_pair


/-- Composing an independent indexed family with an order isomorphism on the elements results in
another independent indexed family. -/
theorem iSupIndep.map_orderIso {ι : Sort*} {α β : Type*} [CompleteLattice α]
    [CompleteLattice β] (f : α ≃o β) {a : ι → α} (ha : iSupIndep a) : iSupIndep (f ∘ a) :=
  fun i => ((ha i).map_orderIso f).mono_right (f.monotone.le_map_iSup₂ _)


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.map_orderIso := iSupIndep.map_orderIso


@[simp]
theorem iSupIndep_map_orderIso_iff {ι : Sort*} {α β : Type*} [CompleteLattice α]
    [CompleteLattice β] (f : α ≃o β) {a : ι → α} : iSupIndep (f ∘ a) ↔ iSupIndep a :=
  ⟨fun h =>
    have hf : f.symm ∘ f ∘ a = a := congr_arg (· ∘ a) f.left_inv.comp_eq_id
    hf ▸ h.map_orderIso f.symm,
    fun h => h.map_orderIso f⟩


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.independent_map_orderIso_iff := iSupIndep_map_orderIso_iff


/-- If the elements of a set are independent, then any element is disjoint from the `iSup` of some
subset of the rest. -/
theorem iSupIndep.disjoint_biSup {ι : Type*} {α : Type*} [CompleteLattice α] {t : ι → α}
    (ht : iSupIndep t) {x : ι} {y : Set ι} (hx : x ∉ y) : Disjoint (t x) (⨆ i ∈ y, t i) :=
  Disjoint.mono_right (biSup_mono fun _ hi => (ne_of_mem_of_not_mem hi hx : _)) (ht x)


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.disjoint_biSup := iSupIndep.disjoint_biSup


lemma iSupIndep.of_coe_Iic_comp {ι : Sort*} {a : α} {t : ι → Set.Iic a}
    (ht : iSupIndep ((↑) ∘ t : ι → α)) : iSupIndep t := by
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    a : α
    t : ι → ↑(Set.Iic a)
    ht : iSupIndep (Function.comp Subtype.val t)
    ⊢ iSupIndep t
  -/
  intro i x
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    a : α
    t : ι → ↑(Set.Iic a)
    ht : iSupIndep (Function.comp Subtype.val t)
    i : ι
    x : ↑(Set.Iic a)
    ⊢ LE.le x (t i) → LE.le x (iSup fun j => iSup fun x => t j) → LE.le x Bot.bot
  -/
  specialize ht i
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    a : α
    t : ι → ↑(Set.Iic a)
    i : ι
    x : ↑(Set.Iic a)
    ht : Disjoint (Function.comp Subtype.val t i) (iSup fun j => iSup fun x => Fun …
    ⊢ LE.le x (t i) → LE.le x (iSup fun j => iSup fun x => t j) → LE.le x Bot.bot
  -/
  simp_rw [Function.comp_apply, ← Set.Iic.coe_iSup] at ht
  /-
    α : Type u_1
    inst✝ : CompleteLattice α
    ι : Sort u_5
    a : α
    t : ι → ↑(Set.Iic a)
    i : ι
    x : ↑(Set.Iic a)
    ht : Disjoint ↑(t i) ↑(iSup fun i_1 => iSup fun i => t i_1)
    ⊢ LE.le x (t i) → LE.le x (iSup fun j => iSup fun x => t j) → LE.le x Bot.bot
  -/
  exact @ht x
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.independent_of_independent_coe_Iic_comp := iSupIndep.of_coe_Iic_comp


theorem iSupIndep_iff_supIndep {s : Finset ι} {f : ι → α} :
    iSupIndep (f ∘ ((↑) : s → ι)) ↔ s.SupIndep f := by
  classical
    rw [Finset.supIndep_iff_disjoint_erase]
    refine Subtype.forall.trans (forall₂_congr fun a b => ?_)
    rw [Finset.sup_eq_iSup]
    congr! 1
    refine iSup_subtype.trans ?_
    congr! 1
    simp [iSup_and, @iSup_comm _ (_ ∈ s)]


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.independent_iff_supIndep := iSupIndep_iff_supIndep


alias ⟨iSupIndep.supIndep, Finset.SupIndep.independent⟩ := iSupIndep_iff_supIndep


theorem iSupIndep.supIndep' {f : ι → α} (s : Finset ι) (h : iSupIndep f) : s.SupIndep f :=
  iSupIndep.supIndep (h.comp Subtype.coe_injective)


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.Independent.supIndep' := iSupIndep.supIndep'


/-- A variant of `CompleteLattice.iSupIndep_iff_supIndep` for `Fintype`s. -/
theorem iSupIndep_iff_supIndep_univ [Fintype ι] {f : ι → α} :
    iSupIndep f ↔ Finset.univ.SupIndep f := by
  classical
    simp [Finset.supIndep_iff_disjoint_erase, iSupIndep, Finset.sup_eq_iSup]


@[deprecated (since := "2024-11-24")]
alias CompleteLattice.independent_iff_supIndep_univ := iSupIndep_iff_supIndep_univ


alias ⟨iSupIndep.sup_indep_univ, Finset.SupIndep.iSupIndep_of_univ⟩ := iSupIndep_iff_supIndep_univ


theorem sSupIndep_iff_pairwiseDisjoint {s : Set α} : sSupIndep s ↔ s.PairwiseDisjoint id :=
  ⟨sSupIndep.pairwiseDisjoint, fun hs _ hi =>
    disjoint_sSup_iff.2 fun _ hj => hs hi hj.1 <| Ne.symm hj.2⟩


@[deprecated (since := "2024-11-24")]
alias setIndependent_iff_pairwiseDisjoint := sSupIndep_iff_pairwiseDisjoint


alias ⟨_, _root_.Set.PairwiseDisjoint.sSupIndep⟩ := sSupIndep_iff_pairwiseDisjoint


theorem iSupIndep_iff_pairwiseDisjoint {f : ι → α} : iSupIndep f ↔ Pairwise (Disjoint on f) :=
  ⟨iSupIndep.pairwiseDisjoint, fun hs _ =>
    disjoint_iSup_iff.2 fun _ => disjoint_iSup_iff.2 fun hij => hs hij.symm⟩


@[deprecated (since := "2024-11-24")]
alias independent_iff_pairwiseDisjoint := iSupIndep_iff_pairwiseDisjoint


