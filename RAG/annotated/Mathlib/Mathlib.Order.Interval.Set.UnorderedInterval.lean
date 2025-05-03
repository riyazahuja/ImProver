/-- `uIcc a b` is the set of elements lying between `a` and `b`, with `a` and `b` included.
Note that we define it more generally in a lattice as `Set.Icc (a ⊓ b) (a ⊔ b)`. In a product type,
`uIcc` corresponds to the bounding box of the two elements. -/
def uIcc (a b : α) : Set α := Icc (a ⊓ b) (a ⊔ b)

-- Porting note: temporarily remove `scoped[uIcc]` and use `[[]]` instead of `[]` before a
-- workaround is found.
-- Porting note 2 : now `scoped[Interval]` works again.

/-- `[[a, b]]` denotes the set of elements lying between `a` and `b`, inclusive. -/
scoped[Interval] notation "[[" a ", " b "]]" => Set.uIcc a b


@[simp] lemma dual_uIcc (a b : α) : [[toDual a, toDual b]] = ofDual ⁻¹' [[a, b]] :=
  -- Note: needed to hint `(α := α)` after https://github.com/leanprover-community/mathlib4/pull/8386 (elaboration order?)
  dual_Icc (α := α)


@[simp]
                                                        /-
                                                          α : Type u_1
                                                          inst✝ : Lattice α
                                                          a b : α
                                                          h : LE.le a b
                                                          ⊢ Eq (Set.uIcc a b) (Set.Icc a b)
                                                        -/
lemma uIcc_of_le (h : a ≤ b) : [[a, b]] = Icc a b := by rw [uIcc, inf_eq_left.2 h, sup_eq_right.2 h]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
                                                        /-
                                                          α : Type u_1
                                                          inst✝ : Lattice α
                                                          a b : α
                                                          h : LE.le b a
                                                          ⊢ Eq (Set.uIcc a b) (Set.Icc b a)
                                                        -/
lemma uIcc_of_ge (h : b ≤ a) : [[a, b]] = Icc b a := by rw [uIcc, inf_eq_right.2 h, sup_eq_left.2 h]
                                                        /-
                                                          🎉 no goals
                                                        -/


                                                      /-
                                                        α : Type u_1
                                                        inst✝ : Lattice α
                                                        a b : α
                                                        ⊢ Eq (Set.uIcc a b) (Set.uIcc b a)
                                                      -/
lemma uIcc_comm (a b : α) : [[a, b]] = [[b, a]] := by simp_rw [uIcc, inf_comm, sup_comm]
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma uIcc_of_lt (h : a < b) : [[a, b]] = Icc a b := uIcc_of_le h.le

lemma uIcc_of_gt (h : b < a) : [[a, b]] = Icc b a := uIcc_of_ge h.le


                                       /-
                                         α : Type u_1
                                         inst✝ : Lattice α
                                         a : α
                                         ⊢ Eq (Set.uIcc a a) (Singleton.singleton a)
                                       -/
lemma uIcc_self : [[a, a]] = {a} := by simp [uIcc]
                                       /-
                                         🎉 no goals
                                       -/


@[simp] lemma nonempty_uIcc : [[a, b]].Nonempty := nonempty_Icc.2 inf_le_sup


lemma Icc_subset_uIcc : Icc a b ⊆ [[a, b]] := Icc_subset_Icc inf_le_left le_sup_right

lemma Icc_subset_uIcc' : Icc b a ⊆ [[a, b]] := Icc_subset_Icc inf_le_right le_sup_left


@[simp] lemma left_mem_uIcc : a ∈ [[a, b]] := ⟨inf_le_left, le_sup_left⟩

@[simp] lemma right_mem_uIcc : b ∈ [[a, b]] := ⟨inf_le_right, le_sup_right⟩


lemma mem_uIcc_of_le (ha : a ≤ x) (hb : x ≤ b) : x ∈ [[a, b]] := Icc_subset_uIcc ⟨ha, hb⟩

lemma mem_uIcc_of_ge (hb : b ≤ x) (ha : x ≤ a) : x ∈ [[a, b]] := Icc_subset_uIcc' ⟨hb, ha⟩


lemma uIcc_subset_uIcc (h₁ : a₁ ∈ [[a₂, b₂]]) (h₂ : b₁ ∈ [[a₂, b₂]]) :
    [[a₁, b₁]] ⊆ [[a₂, b₂]] :=
  Icc_subset_Icc (le_inf h₁.1 h₂.1) (sup_le h₁.2 h₂.2)


lemma uIcc_subset_Icc (ha : a₁ ∈ Icc a₂ b₂) (hb : b₁ ∈ Icc a₂ b₂) :
    [[a₁, b₁]] ⊆ Icc a₂ b₂ :=
  Icc_subset_Icc (le_inf ha.1 hb.1) (sup_le ha.2 hb.2)


lemma uIcc_subset_uIcc_iff_mem :
    [[a₁, b₁]] ⊆ [[a₂, b₂]] ↔ a₁ ∈ [[a₂, b₂]] ∧ b₁ ∈ [[a₂, b₂]] :=
  Iff.intro (fun h => ⟨h left_mem_uIcc, h right_mem_uIcc⟩) fun h =>
    uIcc_subset_uIcc h.1 h.2


lemma uIcc_subset_uIcc_iff_le' :
    [[a₁, b₁]] ⊆ [[a₂, b₂]] ↔ a₂ ⊓ b₂ ≤ a₁ ⊓ b₁ ∧ a₁ ⊔ b₁ ≤ a₂ ⊔ b₂ :=
  Icc_subset_Icc_iff inf_le_sup


lemma uIcc_subset_uIcc_right (h : x ∈ [[a, b]]) : [[x, b]] ⊆ [[a, b]] :=
  uIcc_subset_uIcc h right_mem_uIcc


lemma uIcc_subset_uIcc_left (h : x ∈ [[a, b]]) : [[a, x]] ⊆ [[a, b]] :=
  uIcc_subset_uIcc left_mem_uIcc h


lemma bdd_below_bdd_above_iff_subset_uIcc (s : Set α) :
    BddBelow s ∧ BddAbove s ↔ ∃ a b, s ⊆ [[a, b]] :=
  bddBelow_bddAbove_iff_subset_Icc.trans
    ⟨fun ⟨a, b, h⟩ => ⟨a, b, fun _ hx => Icc_subset_uIcc (h hx)⟩, fun ⟨_, _, h⟩ => ⟨_, _, h⟩⟩


@[simp]
theorem uIcc_prod_uIcc (a₁ a₂ : α) (b₁ b₂ : β) :
    [[a₁, a₂]] ×ˢ [[b₁, b₂]] = [[(a₁, b₁), (a₂, b₂)]] :=
  Icc_prod_Icc _ _ _ _


                                                                                   /-
                                                                                     α : Type u_1
                                                                                     β : Type u_2
                                                                                     inst✝¹ : Lattice α
                                                                                     inst✝ : Lattice β
                                                                                     a b : Prod α β
                                                                                     ⊢ Eq (Set.uIcc a b) (SProd.sprod (Set.uIcc a.1 b.1) (Set.uIcc a.2 b.2))
                                                                                   -/
theorem uIcc_prod_eq (a b : α × β) : [[a, b]] = [[a.1, b.1]] ×ˢ [[a.2, b.2]] := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


lemma eq_of_mem_uIcc_of_mem_uIcc (ha : a ∈ [[b, c]]) (hb : b ∈ [[a, c]]) : a = b :=
  eq_of_inf_eq_sup_eq (inf_congr_right ha.1 hb.1) <| sup_congr_right ha.2 hb.2


lemma eq_of_mem_uIcc_of_mem_uIcc' : b ∈ [[a, c]] → c ∈ [[a, b]] → b = c := by
  /-
    α : Type u_1
    inst✝ : DistribLattice α
    a b c : α
    ⊢ Membership.mem (Set.uIcc a c) b → Membership.mem (Set.uIcc a b) c → Eq b c
  -/
  simpa only [uIcc_comm a] using eq_of_mem_uIcc_of_mem_uIcc
  /-
    🎉 no goals
  -/


lemma uIcc_injective_right (a : α) : Injective fun b => uIcc b a := fun b c h => by
  /-
    α : Type u_1
    inst✝ : DistribLattice α
    a b c : α
    h : Eq ((fun b => Set.uIcc b a) b) ((fun b => Set.uIcc b a) c)
    ⊢ Eq b c
  -/
  rw [Set.ext_iff] at h
  /-
    α : Type u_1
    inst✝ : DistribLattice α
    a b c : α
    h : ∀ (x : α), Iff (Membership.mem ((fun b => Set.uIcc b a) b) x) (Membership. …
    ⊢ Eq b c
  -/
  exact eq_of_mem_uIcc_of_mem_uIcc ((h _).1 left_mem_uIcc) ((h _).2 left_mem_uIcc)
  /-
    🎉 no goals
  -/


lemma uIcc_injective_left (a : α) : Injective (uIcc a) := by
  /-
    α : Type u_1
    inst✝ : DistribLattice α
    a : α
    ⊢ Function.Injective (Set.uIcc a)
  -/
  simpa only [uIcc_comm] using uIcc_injective_right a
  /-
    🎉 no goals
  -/


lemma _root_.MonotoneOn.mapsTo_uIcc (hf : MonotoneOn f (uIcc a b)) :
    MapsTo f (uIcc a b) (uIcc (f a) (f b)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Lattice β
    f : α → β
    a b : α
    hf : MonotoneOn f (Set.uIcc a b)
    ⊢ Set.MapsTo f (Set.uIcc a b) (Set.uIcc (f a) (f b))
  -/
  rw [uIcc, uIcc, ← hf.map_sup, ← hf.map_inf] <;>
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Lattice β
      f : α → β
      a b : α
      hf : MonotoneOn f (Set.uIcc a b)
      ⊢ Set.MapsTo f (Set.Icc (Min.min a b) (Max.max a b)) (Set.Icc (f (Min.min a b) …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    apply_rules [left_mem_uIcc, right_mem_uIcc, hf.mapsTo_Icc]
    /-
      🎉 no goals
    -/


lemma _root_.AntitoneOn.mapsTo_uIcc (hf : AntitoneOn f (uIcc a b)) :
    MapsTo f (uIcc a b) (uIcc (f a) (f b)) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Lattice β
    f : α → β
    a b : α
    hf : AntitoneOn f (Set.uIcc a b)
    ⊢ Set.MapsTo f (Set.uIcc a b) (Set.uIcc (f a) (f b))
  -/
  rw [uIcc, uIcc, ← hf.map_sup, ← hf.map_inf] <;>
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Lattice β
      f : α → β
      a b : α
      hf : AntitoneOn f (Set.uIcc a b)
      ⊢ Set.MapsTo f (Set.Icc (Min.min a b) (Max.max a b)) (Set.Icc (f (Max.max a b) …
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    apply_rules [left_mem_uIcc, right_mem_uIcc, hf.mapsTo_Icc]
    /-
      🎉 no goals
    -/


lemma _root_.Monotone.mapsTo_uIcc (hf : Monotone f) : MapsTo f (uIcc a b) (uIcc (f a) (f b)) :=
  (hf.monotoneOn _).mapsTo_uIcc


lemma _root_.Antitone.mapsTo_uIcc (hf : Antitone f) : MapsTo f (uIcc a b) (uIcc (f a) (f b)) :=
  (hf.antitoneOn _).mapsTo_uIcc


lemma _root_.MonotoneOn.image_uIcc_subset (hf : MonotoneOn f (uIcc a b)) :
    f '' uIcc a b ⊆ uIcc (f a) (f b) := hf.mapsTo_uIcc.image_subset


lemma _root_.AntitoneOn.image_uIcc_subset (hf : AntitoneOn f (uIcc a b)) :
    f '' uIcc a b ⊆ uIcc (f a) (f b) := hf.mapsTo_uIcc.image_subset


lemma _root_.Monotone.image_uIcc_subset (hf : Monotone f) : f '' uIcc a b ⊆ uIcc (f a) (f b) :=
  (hf.monotoneOn _).image_uIcc_subset


lemma _root_.Antitone.image_uIcc_subset (hf : Antitone f) : f '' uIcc a b ⊆ uIcc (f a) (f b) :=
  (hf.antitoneOn _).image_uIcc_subset


theorem Icc_min_max : Icc (min a b) (max a b) = [[a, b]] :=
  rfl


lemma uIcc_of_not_le (h : ¬a ≤ b) : [[a, b]] = Icc b a := uIcc_of_gt <| lt_of_not_ge h

lemma uIcc_of_not_ge (h : ¬b ≤ a) : [[a, b]] = Icc a b := uIcc_of_lt <| lt_of_not_ge h


                                                         /-
                                                           α : Type u_1
                                                           inst✝ : LinearOrder α
                                                           a b : α
                                                           ⊢ Eq (Set.uIcc a b) (Union.union (Set.Icc a b) (Set.Icc b a))
                                                         -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
lemma uIcc_eq_union : [[a, b]] = Icc a b ∪ Icc b a := by rw [Icc_union_Icc', max_comm] <;> rfl
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


                                                                    /-
                                                                      α : Type u_1
                                                                      inst✝ : LinearOrder α
                                                                      a b c : α
                                                                      ⊢ Iff (Membership.mem (Set.uIcc b c) a) (Or (And (LE.le b a) (LE.le a c)) (And …
                                                                    -/
lemma mem_uIcc : a ∈ [[b, c]] ↔ b ≤ a ∧ a ≤ c ∨ c ≤ a ∧ a ≤ b := by simp [uIcc_eq_union]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


lemma not_mem_uIcc_of_lt (ha : c < a) (hb : c < b) : c ∉ [[a, b]] :=
  not_mem_Icc_of_lt <| lt_min_iff.mpr ⟨ha, hb⟩


lemma not_mem_uIcc_of_gt (ha : a < c) (hb : b < c) : c ∉ [[a, b]] :=
  not_mem_Icc_of_gt <| max_lt_iff.mpr ⟨ha, hb⟩


lemma uIcc_subset_uIcc_iff_le :
    [[a₁, b₁]] ⊆ [[a₂, b₂]] ↔ min a₂ b₂ ≤ min a₁ b₁ ∧ max a₁ b₁ ≤ max a₂ b₂ :=
  uIcc_subset_uIcc_iff_le'


/-- A sort of triangle inequality. -/
lemma uIcc_subset_uIcc_union_uIcc : [[a, c]] ⊆ [[a, b]] ∪ [[b, c]] := fun x => by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c x : α
    ⊢ Membership.mem (Set.uIcc a c) x → Membership.mem (Union.union (Set.uIcc a b) …
  -/
  simp only [mem_uIcc, mem_union]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c x : α
    ⊢ Or (And (LE.le a x) (LE.le x c)) (And (LE.le c x) (LE.le x a)) → Or (Or (And …
  -/
                                       /-
                                         🎉 no goals
                                       -/
  rcases le_total x b with h2 | h2 <;> tauto
                                       /-
                                         🎉 no goals
                                       -/


lemma monotone_or_antitone_iff_uIcc :
    Monotone f ∨ Antitone f ↔ ∀ a b c, c ∈ [[a, b]] → f c ∈ [[f a, f b]] := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ Iff (Or (Monotone f) (Antitone f)) (∀ (a b c : α), Membership.mem (Set.uIcc  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      ⊢ Or (Monotone f) (Antitone f) → ∀ (a b c : α), Membership.mem (Set.uIcc a b)  …
    -/
  · rintro (hf | hf) a b c <;> simp_rw [← Icc_min_max, ← hf.map_min, ← hf.map_max]
    /-
      case mp.inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      hf : Monotone f
      a b c : α
      ⊢ Membership.mem (Set.Icc (Min.min a b) (Max.max a b)) c → Membership.mem (Set …
    -/
    exacts [fun hc => ⟨hf hc.1, hf hc.2⟩, fun hc => ⟨hf hc.2, hf hc.1⟩]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ (∀ (a b c : α), Membership.mem (Set.uIcc a b) c → Membership.mem (Set.uIcc ( …
  -/
  contrapose!
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ And (Not (Monotone f)) (Not (Antitone f)) → Exists fun a => Exists fun b =>  …
  -/
  rw [not_monotone_not_antitone_iff_exists_le_le]
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : LinearOrder β
    f : α → β
    ⊢ (Exists fun a => Exists fun b => Exists fun c => And (LE.le a b) (And (LE.le …
  -/
  rintro ⟨a, b, c, hab, hbc, ⟨hfab, hfcb⟩ | ⟨hfba, hfbc⟩⟩
    /-
      case mpr.intro.intro.intro.intro.intro.inl.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b c : α
      hab : LE.le a b
      hbc : LE.le b c
      hfab : LT.lt (f a) (f b)
      hfcb : LT.lt (f c) (f b)
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (Membership.mem (Set.uIc …
    -/
  · exact ⟨a, c, b, Icc_subset_uIcc ⟨hab, hbc⟩, fun h => h.2.not_lt <| max_lt hfab hfcb⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro.intro.intro.intro.intro.inr.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : LinearOrder β
      f : α → β
      a b c : α
      hab : LE.le a b
      hbc : LE.le b c
      hfba : LT.lt (f b) (f a)
      hfbc : LT.lt (f b) (f c)
      ⊢ Exists fun a => Exists fun b => Exists fun c => And (Membership.mem (Set.uIc …
    -/
  · exact ⟨a, c, b, Icc_subset_uIcc ⟨hab, hbc⟩, fun h => h.1.not_lt <| lt_min hfba hfbc⟩
    /-
      🎉 no goals
    -/

-- Porting note: mathport expands the syntactic sugar `∀ a b c ∈ s` differently than Lean3

lemma monotoneOn_or_antitoneOn_iff_uIcc :
    MonotoneOn f s ∨ AntitoneOn f s ↔
      ∀ᵉ (a ∈ s) (b ∈ s) (c ∈ s), c ∈ [[a, b]] → f c ∈ [[f a, f b]] := by
  simp [monotoneOn_iff_monotone, antitoneOn_iff_antitone, monotone_or_antitone_iff_uIcc,
    mem_uIcc]

-- Porting note: what should the naming scheme be here? This is a term, so should be `uIoc`,
-- but we also want to match the `Ioc` convention.

/-- The open-closed uIcc with unordered bounds. -/
def uIoc : α → α → Set α := fun a b => Ioc (min a b) (max a b)

-- Porting note: removed `scoped[uIcc]` temporarily before a workaround is found
-- Below is a capital iota

/-- `Ι a b` denotes the open-closed interval with unordered bounds. Here, `Ι` is a capital iota,
distinguished from a capital `i`. -/
notation "Ι" => Set.uIoc


                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrder α
                                                               a b : α
                                                               h : LE.le a b
                                                               ⊢ Eq (Set.uIoc a b) (Set.Ioc a b)
                                                             -/
@[simp] lemma uIoc_of_le (h : a ≤ b) : Ι a b = Ioc a b := by simp [uIoc, h]
                                                             /-
                                                               🎉 no goals
                                                             -/

                                                             /-
                                                               α : Type u_1
                                                               inst✝ : LinearOrder α
                                                               a b : α
                                                               h : LE.le b a
                                                               ⊢ Eq (Set.uIoc a b) (Set.Ioc b a)
                                                             -/
@[simp] lemma uIoc_of_ge (h : b ≤ a) : Ι a b = Ioc b a := by simp [uIoc, h]
                                                             /-
                                                               🎉 no goals
                                                             -/


lemma uIoc_eq_union : Ι a b = Ioc a b ∪ Ioc b a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    ⊢ Eq (Set.uIoc a b) (Union.union (Set.Ioc a b) (Set.Ioc b a))
  -/
                         /-
                           🎉 no goals
                         -/
  cases le_total a b <;> simp [uIoc, *]
                         /-
                           🎉 no goals
                         -/


lemma mem_uIoc : a ∈ Ι b c ↔ b < a ∧ a ≤ c ∨ c < a ∧ a ≤ b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Iff (Membership.mem (Set.uIoc b c) a) (Or (And (LT.lt b a) (LE.le a c)) (And …
  -/
  rw [uIoc_eq_union, mem_union, mem_Ioc, mem_Ioc]
  /-
    🎉 no goals
  -/


lemma not_mem_uIoc : a ∉ Ι b c ↔ a ≤ b ∧ a ≤ c ∨ c < a ∧ b < a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Iff (Not (Membership.mem (Set.uIoc b c) a)) (Or (And (LE.le a b) (LE.le a c) …
  -/
  simp only [uIoc_eq_union, mem_union, mem_Ioc, not_lt, ← not_le]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Iff (Not (Or (And (Not (LE.le a b)) (LE.le a c)) (And (Not (LE.le a c)) (LE. …
  -/
  tauto
  /-
    🎉 no goals
  -/


                                                      /-
                                                        α : Type u_1
                                                        inst✝ : LinearOrder α
                                                        a b : α
                                                        ⊢ Iff (Membership.mem (Set.uIoc a b) a) (LT.lt b a)
                                                      -/
@[simp] lemma left_mem_uIoc : a ∈ Ι a b ↔ b < a := by simp [mem_uIoc]
                                                      /-
                                                        🎉 no goals
                                                      -/

                                                       /-
                                                         α : Type u_1
                                                         inst✝ : LinearOrder α
                                                         a b : α
                                                         ⊢ Iff (Membership.mem (Set.uIoc a b) b) (LT.lt a b)
                                                       -/
@[simp] lemma right_mem_uIoc : b ∈ Ι a b ↔ a < b := by simp [mem_uIoc]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma forall_uIoc_iff {P : α → Prop} :
    (∀ x ∈ Ι a b, P x) ↔ (∀ x ∈ Ioc a b, P x) ∧ ∀ x ∈ Ioc b a, P x := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    P : α → Prop
    ⊢ Iff (∀ (x : α), Membership.mem (Set.uIoc a b) x → P x) (And (∀ (x : α), Memb …
  -/
  simp only [uIoc_eq_union, mem_union, or_imp, forall_and]
  /-
    🎉 no goals
  -/


lemma uIoc_subset_uIoc_of_uIcc_subset_uIcc {a b c d : α}
    (h : [[a, b]] ⊆ [[c, d]]) : Ι a b ⊆ Ι c d :=
  Ioc_subset_Ioc (uIcc_subset_uIcc_iff_le.1 h).1 (uIcc_subset_uIcc_iff_le.1 h).2


                                                /-
                                                  α : Type u_1
                                                  inst✝ : LinearOrder α
                                                  a b : α
                                                  ⊢ Eq (Set.uIoc a b) (Set.uIoc b a)
                                                -/
lemma uIoc_comm (a b : α) : Ι a b = Ι b a := by simp only [uIoc, min_comm a b, max_comm a b]
                                                /-
                                                  🎉 no goals
                                                -/


lemma Ioc_subset_uIoc : Ioc a b ⊆ Ι a b := Ioc_subset_Ioc (min_le_left _ _) (le_max_right _ _)

lemma Ioc_subset_uIoc' : Ioc a b ⊆ Ι b a := Ioc_subset_Ioc (min_le_right _ _) (le_max_left _ _)


lemma uIoc_subset_uIcc : Ι a b ⊆ uIcc a b := Ioc_subset_Icc_self


lemma eq_of_mem_uIoc_of_mem_uIoc : a ∈ Ι b c → b ∈ Ι a c → a = b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Membership.mem (Set.uIoc b c) a → Membership.mem (Set.uIoc a c) b → Eq a b
  -/
  simp_rw [mem_uIoc]; rintro (⟨_, _⟩ | ⟨_, _⟩) (⟨_, _⟩ | ⟨_, _⟩) <;> apply le_antisymm <;>
    /-
      case inl.intro.inl.intro.a
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      left✝¹ : LT.lt b a
      right✝¹ : LE.le a c
      left✝ : LT.lt a b
      right✝ : LE.le b c
      ⊢ LE.le a b
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    first |assumption|exact le_of_lt ‹_›|exact le_trans ‹_› (le_of_lt ‹_›)
    /-
      🎉 no goals
    -/


lemma eq_of_mem_uIoc_of_mem_uIoc' : b ∈ Ι a c → c ∈ Ι a b → b = c := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ⊢ Membership.mem (Set.uIoc a c) b → Membership.mem (Set.uIoc a b) c → Eq b c
  -/
  simpa only [uIoc_comm a] using eq_of_mem_uIoc_of_mem_uIoc
  /-
    🎉 no goals
  -/


lemma eq_of_not_mem_uIoc_of_not_mem_uIoc (ha : a ≤ c) (hb : b ≤ c) :
    a ∉ Ι b c → b ∉ Ι a c → a = b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ha : LE.le a c
    hb : LE.le b c
    ⊢ Not (Membership.mem (Set.uIoc b c) a) → Not (Membership.mem (Set.uIoc a c) b …
  -/
  simp_rw [not_mem_uIoc]
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    ha : LE.le a c
    hb : LE.le b c
    ⊢ Or (And (LE.le a b) (LE.le a c)) (And (LT.lt c a) (LT.lt b a)) → Or (And (LE …
  -/
  rintro (⟨_, _⟩ | ⟨_, _⟩) (⟨_, _⟩ | ⟨_, _⟩) <;>
      /-
        case inl.intro.inl.intro
        α : Type u_1
        inst✝ : LinearOrder α
        a b c : α
        ha : LE.le a c
        hb : LE.le b c
        left✝¹ : LE.le a b
        right✝¹ : LE.le a c
        left✝ : LE.le b a
        right✝ : LE.le b c
        ⊢ Eq a b
      -/
      apply le_antisymm <;>
    first |assumption|exact le_of_lt ‹_›|
    exact absurd hb (not_le_of_lt ‹c < b›)|exact absurd ha (not_le_of_lt ‹c < a›)


lemma uIoc_injective_right (a : α) : Injective fun b => Ι b a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a : α
    ⊢ Function.Injective fun b => Set.uIoc b a
  -/
  rintro b c h
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    h : Eq ((fun b => Set.uIoc b a) b) ((fun b => Set.uIoc b a) c)
    ⊢ Eq b c
  -/
  rw [Set.ext_iff] at h
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b c : α
    h : ∀ (x : α), Iff (Membership.mem ((fun b => Set.uIoc b a) b) x) (Membership. …
    ⊢ Eq b c
  -/
  obtain ha | ha := le_or_lt b a
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      h : ∀ (x : α), Iff (Membership.mem ((fun b => Set.uIoc b a) b) x) (Membership. …
      ha : LE.le b a
      ⊢ Eq b c
    -/
  · have hb := (h b).not
    simp only [ha, left_mem_uIoc, not_lt, true_iff, not_mem_uIoc, ← not_le,
      and_true, not_true, false_and, not_false_iff, or_false] at hb
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      h : ∀ (x : α), Iff (Membership.mem ((fun b => Set.uIoc b a) b) x) (Membership. …
      ha : LE.le b a
      hb : LE.le b c
      ⊢ Eq b c
    -/
    refine hb.eq_of_not_lt fun hc => ?_
    /-
      case inl
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      h : ∀ (x : α), Iff (Membership.mem ((fun b => Set.uIoc b a) b) x) (Membership. …
      ha : LE.le b a
      hb : LE.le b c
      hc : LT.lt b c
      ⊢ False
    -/
    simpa [ha, and_iff_right hc, ← @not_le _ _ _ a, iff_not_self, -not_le] using h c
    /-
      🎉 no goals
    -/
  · refine
      eq_of_mem_uIoc_of_mem_uIoc ((h _).1 <| left_mem_uIoc.2 ha)
        ((h _).2 <| left_mem_uIoc.2 <| ha.trans_le ?_)
    /-
      case inr
      α : Type u_1
      inst✝ : LinearOrder α
      a b c : α
      h : ∀ (x : α), Iff (Membership.mem ((fun b => Set.uIoc b a) b) x) (Membership. …
      ha : LT.lt a b
      ⊢ LE.le b c
    -/
    simpa [ha, ha.not_le, mem_uIoc] using h b
    /-
      🎉 no goals
    -/


lemma uIoc_injective_left (a : α) : Injective (Ι a) := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a : α
    ⊢ Function.Injective (Set.uIoc a)
  -/
  simpa only [uIoc_comm] using uIoc_injective_right a
  /-
    🎉 no goals
  -/


/-- `uIoo a b` is the set of elements lying between `a` and `b`, with `a` and `b` not included.
Note that we define it more generally in a lattice as `Set.Ioo (a ⊓ b) (a ⊔ b)`. In a product type,
`uIoo` corresponds to the bounding box of the two elements. -/
def uIoo (a b : α) : Set α := Ioo (a ⊓ b) (a ⊔ b)


@[simp] lemma dual_uIoo (a b : α) : uIoo (toDual a) (toDual b) = ofDual ⁻¹' uIoo a b :=
  dual_Ioo (α := α)


@[simp] lemma uIoo_of_le (h : a ≤ b) : uIoo a b = Ioo a b := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le a b
    ⊢ Eq (Set.uIoo a b) (Set.Ioo a b)
  -/
  rw [uIoo, inf_eq_left.2 h, sup_eq_right.2 h]
  /-
    🎉 no goals
  -/


@[simp] lemma uIoo_of_ge (h : b ≤ a) : uIoo a b = Ioo b a := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    a b : α
    h : LE.le b a
    ⊢ Eq (Set.uIoo a b) (Set.Ioo b a)
  -/
  rw [uIoo, inf_eq_right.2 h, sup_eq_left.2 h]
  /-
    🎉 no goals
  -/


                                                      /-
                                                        α : Type u_1
                                                        inst✝ : LinearOrder α
                                                        a b : α
                                                        ⊢ Eq (Set.uIoo a b) (Set.uIoo b a)
                                                      -/
lemma uIoo_comm (a b : α) : uIoo a b = uIoo b a := by simp_rw [uIoo, inf_comm, sup_comm]
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma uIoo_of_lt (h : a < b) : uIoo a b = Ioo a b := uIoo_of_le h.le


lemma uIoo_of_gt (h : b < a) : uIoo a b = Ioo b a := uIoo_of_ge h.le


                                     /-
                                       α : Type u_1
                                       inst✝ : LinearOrder α
                                       a : α
                                       ⊢ Eq (Set.uIoo a a) EmptyCollection.emptyCollection
                                     -/
lemma uIoo_self : uIoo a a = ∅ := by simp [uIoo]
                                     /-
                                       🎉 no goals
                                     -/


lemma Ioo_subset_uIoo : Ioo a b ⊆ uIoo a b := Ioo_subset_Ioo inf_le_left le_sup_right


/-- Same as `Ioo_subset_uIoo` but with `Ioo a b` replaced by `Ioo b a`. -/
lemma Ioo_subset_uIoo' : Ioo b a ⊆ uIoo a b := Ioo_subset_Ioo inf_le_right le_sup_left


lemma mem_uIoo_of_lt (ha : a < x) (hb : x < b) : x ∈ uIoo a b := Ioo_subset_uIoo ⟨ha, hb⟩


lemma mem_uIoo_of_gt (hb : b < x) (ha : x < a) : x ∈ uIoo a b := Ioo_subset_uIoo' ⟨hb, ha⟩


theorem Ioo_min_max : Ioo (min a b) (max a b) = uIoo a b := rfl


lemma uIoo_of_not_le (h : ¬a ≤ b) : uIoo a b = Ioo b a := uIoo_of_gt <| lt_of_not_ge h


lemma uIoo_of_not_ge (h : ¬b ≤ a) : uIoo a b = Ioo a b := uIoo_of_lt <| lt_of_not_ge h


theorem uIoo_subset_uIcc {α : Type*} [LinearOrder α] (a : α) (b : α) : uIoo a b ⊆ uIcc a b := by
  /-
    α : Type u_3
    inst✝ : LinearOrder α
    a b : α
    ⊢ HasSubset.Subset (Set.uIoo a b) (Set.uIcc a b)
  -/
  simp [uIoo, uIcc, Ioo_subset_Icc_self]
  /-
    🎉 no goals
  -/


