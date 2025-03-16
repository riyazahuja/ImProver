/-- `DFinsupp.Lex r s` is the lexicographic relation on `Π₀ i, α i`, where `ι` is ordered by `r`,
and `α i` is ordered by `s i`.
The type synonym `Lex (Π₀ i, α i)` has an order given by `DFinsupp.Lex (· < ·) (· < ·)`.
-/
protected def Lex (r : ι → ι → Prop) (s : ∀ i, α i → α i → Prop) (x y : Π₀ i, α i) : Prop :=
  Pi.Lex r (s _) x y

-- Porting note: Added `_root_` to match more closely with Lean 3. Also updated `s`'s type.

theorem _root_.Pi.lex_eq_dfinsupp_lex {r : ι → ι → Prop} {s : ∀ i, α i → α i → Prop}
    (a b : Π₀ i, α i) : Pi.Lex r (s _) (a : ∀ i, α i) b = DFinsupp.Lex r s a b :=
  rfl

-- Porting note: Updated `s`'s type.

theorem lex_def {r : ι → ι → Prop} {s : ∀ i, α i → α i → Prop} {a b : Π₀ i, α i} :
    DFinsupp.Lex r s a b ↔ ∃ j, (∀ d, r d j → a d = b d) ∧ s j (a j) (b j) :=
  Iff.rfl


instance [LT ι] [∀ i, LT (α i)] : LT (Lex (Π₀ i, α i)) :=
  ⟨fun f g ↦ DFinsupp.Lex (· < ·) (fun _ ↦ (· < ·)) (ofLex f) (ofLex g)⟩


theorem lex_lt_of_lt_of_preorder [∀ i, Preorder (α i)] (r) [IsStrictOrder ι r] {x y : Π₀ i, α i}
    (hlt : x < y) : ∃ i, (∀ j, r j i → x j ≤ y j ∧ y j ≤ x j) ∧ x i < y i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    inst✝¹ : (i : ι) → Preorder (α i)
    r : ι → ι → Prop
    inst✝ : IsStrictOrder ι r
    x y : DFinsupp fun i => α i
    hlt : LT.lt x y
    ⊢ Exists fun i => And (∀ (j : ι), r j i → And (LE.le (x j) (y j)) (LE.le (y j) …
  -/
  obtain ⟨hle, j, hlt⟩ := Pi.lt_def.1 hlt
  classical
  have : (x.neLocus y : Set ι).WellFoundedOn r := (x.neLocus y).finite_toSet.wellFoundedOn
  obtain ⟨i, hi, hl⟩ := this.has_min { i | x i < y i } ⟨⟨j, mem_neLocus.2 hlt.ne⟩, hlt⟩
  refine ⟨i, fun k hk ↦ ⟨hle k, ?_⟩, hi⟩
  exact of_not_not fun h ↦ hl ⟨k, mem_neLocus.2 (ne_of_not_le h).symm⟩ ((hle k).lt_of_not_le h) hk


theorem lex_lt_of_lt [∀ i, PartialOrder (α i)] (r) [IsStrictOrder ι r] {x y : Π₀ i, α i}
    (hlt : x < y) : Pi.Lex r (· < ·) x y := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    inst✝¹ : (i : ι) → PartialOrder (α i)
    r : ι → ι → Prop
    inst✝ : IsStrictOrder ι r
    x y : DFinsupp fun i => α i
    hlt : LT.lt x y
    ⊢ Pi.Lex r (fun {i} x1 x2 => LT.lt x1 x2) ⇑x ⇑y
  -/
  simp_rw [Pi.Lex, le_antisymm_iff]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    inst✝¹ : (i : ι) → PartialOrder (α i)
    r : ι → ι → Prop
    inst✝ : IsStrictOrder ι r
    x y : DFinsupp fun i => α i
    hlt : LT.lt x y
    ⊢ Exists fun i => And (∀ (j : ι), r j i → And (LE.le (x j) (y j)) (LE.le (y j) …
  -/
  exact lex_lt_of_lt_of_preorder r hlt
  /-
    🎉 no goals
  -/


instance Lex.isStrictOrder [∀ i, PartialOrder (α i)] :
    IsStrictOrder (Lex (Π₀ i, α i)) (· < ·) where
  irrefl _ := lt_irrefl (α := Lex (∀ i, α i)) _
  trans _ _ _ := lt_trans (α := Lex (∀ i, α i))


/-- The partial order on `DFinsupp`s obtained by the lexicographic ordering.
See `DFinsupp.Lex.linearOrder` for a proof that this partial order is in fact linear. -/
instance Lex.partialOrder [∀ i, PartialOrder (α i)] : PartialOrder (Lex (Π₀ i, α i)) where
  lt := (· < ·)
  le x y := ⇑(ofLex x) = ⇑(ofLex y) ∨ x < y
  __ := PartialOrder.lift (fun x : Lex (Π₀ i, α i) ↦ toLex (⇑(ofLex x)))
    (DFunLike.coe_injective (F := DFinsupp α))


/-- Auxiliary helper to case split computably. There is no need for this to be public, as it
can be written with `Or.by_cases` on `lt_trichotomy` once the instances below are constructed. -/
private def lt_trichotomy_rec {P : Lex (Π₀ i, α i) → Lex (Π₀ i, α i) → Sort*}
    (h_lt : ∀ {f g}, toLex f < toLex g → P (toLex f) (toLex g))
    (h_eq : ∀ {f g}, toLex f = toLex g → P (toLex f) (toLex g))
    (h_gt : ∀ {f g}, toLex g < toLex f → P (toLex f) (toLex g)) : ∀ f g, P f g :=
  Lex.rec fun f ↦ Lex.rec fun g ↦ match (motive := ∀ y, (f.neLocus g).min = y → _) _, rfl with
  | ⊤, h => h_eq (neLocus_eq_empty.mp <| Finset.min_eq_top.mp h)
  | (wit : ι), h => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : (i : ι) → Zero (α i)
      inst✝¹ : LinearOrder ι
      inst✝ : (i : ι) → LinearOrder (α i)
      P : Lex (DFinsupp fun i => α i) → Lex (DFinsupp fun i => α i) → Sort u_3
      h_lt : {f g : DFinsupp fun i => α i} → LT.lt (toLex f) (toLex g) → P (toLex f) …
      h_eq : {f g : DFinsupp fun i => α i} → Eq (toLex f) (toLex g) → P (toLex f) (t …
      h_gt : {f g : DFinsupp fun i => α i} → LT.lt (toLex g) (toLex f) → P (toLex f) …
      f g : DFinsupp fun i => α i
      wit : ι
      h : Eq (f.neLocus g).min ↑wit
      ⊢ P (toLex f) (toLex g)
    -/
    apply (mem_neLocus.mp <| Finset.mem_of_min h).lt_or_lt.by_cases <;> intro hwit
      /-
        case h₁
        ι : Type u_1
        α : ι → Type u_2
        inst✝² : (i : ι) → Zero (α i)
        inst✝¹ : LinearOrder ι
        inst✝ : (i : ι) → LinearOrder (α i)
        P : Lex (DFinsupp fun i => α i) → Lex (DFinsupp fun i => α i) → Sort u_3
        h_lt : {f g : DFinsupp fun i => α i} → LT.lt (toLex f) (toLex g) → P (toLex f) …
        h_eq : {f g : DFinsupp fun i => α i} → Eq (toLex f) (toLex g) → P (toLex f) (t …
        h_gt : {f g : DFinsupp fun i => α i} → LT.lt (toLex g) (toLex f) → P (toLex f) …
        f g : DFinsupp fun i => α i
        wit : ι
        h : Eq (f.neLocus g).min ↑wit
        hwit : LT.lt (f wit) (g wit)
        ⊢ P (toLex f) (toLex g)
      -/
    · exact h_lt ⟨wit, fun j hj ↦ not_mem_neLocus.mp (Finset.not_mem_of_lt_min hj h), hwit⟩
      /-
        🎉 no goals
      -/
    · exact h_gt ⟨wit, fun j hj ↦
        not_mem_neLocus.mp (Finset.not_mem_of_lt_min hj <| by rwa [neLocus_comm]), hwit⟩


/-- The less-or-equal relation for the lexicographic ordering is decidable. -/
irreducible_def Lex.decidableLE : DecidableRel (α := Lex (Π₀ i, α i)) (· ≤ ·) :=
  lt_trichotomy_rec (fun h ↦ isTrue <| Or.inr h)
    (fun h ↦ isTrue <| Or.inl <| congr_arg _ h)
    fun h ↦ isFalse fun h' ↦ lt_irrefl _ (h.trans_le h')


/-- The less-than relation for the lexicographic ordering is decidable. -/
irreducible_def Lex.decidableLT : DecidableRel (α := Lex (Π₀ i, α i)) (· < ·) :=
  lt_trichotomy_rec (fun h ↦ isTrue h) (fun h ↦ isFalse h.not_lt) fun h ↦ isFalse h.asymm

-- Porting note: Added `DecidableEq` for `LinearOrder`.

instance : DecidableEq (Lex (Π₀ i, α i)) :=
  lt_trichotomy_rec (fun h ↦ isFalse fun h' ↦ h'.not_lt h) isTrue
    fun h ↦ isFalse fun h' ↦ h'.symm.not_lt h


/-- The linear order on `DFinsupp`s obtained by the lexicographic ordering. -/
instance Lex.linearOrder : LinearOrder (Lex (Π₀ i, α i)) where
  __ := Lex.partialOrder
  le_total := lt_trichotomy_rec (fun h ↦ Or.inl h.le) (fun h ↦ Or.inl h.le) fun h ↦ Or.inr h.le
  decidableLT := decidableLT
  decidableLE := decidableLE
  decidableEq := inferInstance


theorem toLex_monotone : Monotone (@toLex (Π₀ i, α i)) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    inst✝¹ : LinearOrder ι
    inst✝ : (i : ι) → PartialOrder (α i)
    ⊢ Monotone ⇑toLex
  -/
  intro a b h
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : (i : ι) → Zero (α i)
    inst✝¹ : LinearOrder ι
    inst✝ : (i : ι) → PartialOrder (α i)
    a b : DFinsupp fun i => α i
    h : LE.le a b
    ⊢ LE.le (toLex a) (toLex b)
  -/
  refine le_of_lt_or_eq (or_iff_not_imp_right.2 fun hne ↦ ?_)
  classical
  exact ⟨Finset.min' _ (nonempty_neLocus_iff.2 hne),
    fun j hj ↦ not_mem_neLocus.1 fun h ↦ (Finset.min'_le _ _ h).not_lt hj,
    (h _).lt_of_ne (mem_neLocus.1 <| Finset.min'_mem _ _)⟩


theorem lt_of_forall_lt_of_lt (a b : Lex (Π₀ i, α i)) (i : ι) :
    (∀ j < i, ofLex a j = ofLex b j) → ofLex a i < ofLex b i → a < b :=
  fun h1 h2 ↦ ⟨i, h1, h2⟩


instance Lex.addLeftStrictMono : AddLeftStrictMono (Lex (Π₀ i, α i)) :=
  ⟨fun _ _ _ ⟨a, lta, ha⟩ ↦ ⟨a, fun j ja ↦ congr_arg _ (lta j ja), add_lt_add_left ha _⟩⟩


instance Lex.addLeftMono : AddLeftMono (Lex (Π₀ i, α i)) :=
  addLeftMono_of_addLeftStrictMono _


instance Lex.addRightStrictMono : AddRightStrictMono (Lex (Π₀ i, α i)) :=
  ⟨fun f _ _ ⟨a, lta, ha⟩ ↦
    ⟨a, fun j ja ↦ congr_arg (· + ofLex f j) (lta j ja), add_lt_add_right ha _⟩⟩


instance Lex.addRightMono : AddRightMono (Lex (Π₀ i, α i)) :=
  addRightMono_of_addRightStrictMono _


instance Lex.orderBot [∀ i, CanonicallyOrderedAddCommMonoid (α i)] :
    OrderBot (Lex (Π₀ i, α i)) where
  bot := 0
  bot_le _ := DFinsupp.toLex_monotone bot_le


instance Lex.orderedAddCancelCommMonoid [∀ i, OrderedCancelAddCommMonoid (α i)] :
    OrderedCancelAddCommMonoid (Lex (Π₀ i, α i)) where
  add_le_add_left _ _ h _ := add_le_add_left (α := Lex (∀ i, α i)) h _
  le_of_add_le_add_left _ _ _ := le_of_add_le_add_left (α := Lex (∀ i, α i))


instance Lex.orderedAddCommGroup [∀ i, OrderedAddCommGroup (α i)] :
    OrderedAddCommGroup (Lex (Π₀ i, α i)) where
  add_le_add_left _ _ := add_le_add_left


instance Lex.linearOrderedCancelAddCommMonoid
    [∀ i, LinearOrderedCancelAddCommMonoid (α i)] :
    LinearOrderedCancelAddCommMonoid (Lex (Π₀ i, α i)) where
  __ : LinearOrder (Lex (Π₀ i, α i)) := inferInstance
  __ : OrderedCancelAddCommMonoid (Lex (Π₀ i, α i)) := inferInstance


instance Lex.linearOrderedAddCommGroup [∀ i, LinearOrderedAddCommGroup (α i)] :
    LinearOrderedAddCommGroup (Lex (Π₀ i, α i)) where
  __ : LinearOrder (Lex (Π₀ i, α i)) := inferInstance
  add_le_add_left _ _ := add_le_add_left


