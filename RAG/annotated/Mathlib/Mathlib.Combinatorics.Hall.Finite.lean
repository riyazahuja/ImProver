theorem hall_cond_of_erase {x : ι} (a : α)
    (ha : ∀ s : Finset ι, s.Nonempty → s ≠ univ → #s < #(s.biUnion t))
    (s' : Finset { x' : ι | x' ≠ x }) : #s' ≤ #(s'.biUnion fun x' => (t x').erase a) := by
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    x : ι
    a : α
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    s' : Finset ↑(setOf fun x' => Ne x' x)
    ⊢ LE.le s'.card (s'.biUnion fun x' => (t ↑x').erase a).card
  -/
  haveI := Classical.decEq ι
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    x : ι
    a : α
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    s' : Finset ↑(setOf fun x' => Ne x' x)
    this : DecidableEq ι
    ⊢ LE.le s'.card (s'.biUnion fun x' => (t ↑x').erase a).card
  -/
  specialize ha (s'.image fun z => z.1)
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    x : ι
    a : α
    s' : Finset ↑(setOf fun x' => Ne x' x)
    this : DecidableEq ι
    ha : (Finset.image (fun z => ↑z) s').Nonempty → Ne (Finset.image (fun z => ↑z) …
    ⊢ LE.le s'.card (s'.biUnion fun x' => (t ↑x').erase a).card
  -/
  rw [image_nonempty, Finset.card_image_of_injective s' Subtype.coe_injective] at ha
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    x : ι
    a : α
    s' : Finset ↑(setOf fun x' => Ne x' x)
    this : DecidableEq ι
    ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
    ⊢ LE.le s'.card (s'.biUnion fun x' => (t ↑x').erase a).card
  -/
  by_cases he : s'.Nonempty
  · have ha' : #s' < #(s'.biUnion fun x => t x) := by
      convert ha he fun h => by simpa [← h] using mem_univ x using 2
      ext x
      simp only [mem_image, mem_biUnion, exists_prop, SetCoe.exists, exists_and_right,
        exists_eq_right, Subtype.coe_mk]
    /-
      case pos
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      x : ι
      a : α
      s' : Finset ↑(setOf fun x' => Ne x' x)
      this : DecidableEq ι
      ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
      he : s'.Nonempty
      ha' : LT.lt s'.card (s'.biUnion fun x_1 => t ↑x_1).card
      ⊢ LE.le s'.card (s'.biUnion fun x' => (t ↑x').erase a).card
    -/
    rw [← erase_biUnion]
    /-
      case pos
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      x : ι
      a : α
      s' : Finset ↑(setOf fun x' => Ne x' x)
      this : DecidableEq ι
      ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
      he : s'.Nonempty
      ha' : LT.lt s'.card (s'.biUnion fun x_1 => t ↑x_1).card
      ⊢ LE.le s'.card ((s'.biUnion fun x' => t ↑x').erase a).card
    -/
    by_cases hb : a ∈ s'.biUnion fun x => t x
      /-
        case pos
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        x : ι
        a : α
        s' : Finset ↑(setOf fun x' => Ne x' x)
        this : DecidableEq ι
        ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
        he : s'.Nonempty
        ha' : LT.lt s'.card (s'.biUnion fun x_1 => t ↑x_1).card
        hb : Membership.mem (s'.biUnion fun x_1 => t ↑x_1) a
        ⊢ LE.le s'.card ((s'.biUnion fun x' => t ↑x').erase a).card
      -/
    · rw [card_erase_of_mem hb]
      /-
        case pos
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        x : ι
        a : α
        s' : Finset ↑(setOf fun x' => Ne x' x)
        this : DecidableEq ι
        ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
        he : s'.Nonempty
        ha' : LT.lt s'.card (s'.biUnion fun x_1 => t ↑x_1).card
        hb : Membership.mem (s'.biUnion fun x_1 => t ↑x_1) a
        ⊢ LE.le s'.card (HSub.hSub (s'.biUnion fun x_1 => t ↑x_1).card 1)
      -/
      exact Nat.le_sub_one_of_lt ha'
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        x : ι
        a : α
        s' : Finset ↑(setOf fun x' => Ne x' x)
        this : DecidableEq ι
        ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
        he : s'.Nonempty
        ha' : LT.lt s'.card (s'.biUnion fun x_1 => t ↑x_1).card
        hb : Not (Membership.mem (s'.biUnion fun x_1 => t ↑x_1) a)
        ⊢ LE.le s'.card ((s'.biUnion fun x' => t ↑x').erase a).card
      -/
    · rw [erase_eq_of_not_mem hb]
      /-
        case neg
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        x : ι
        a : α
        s' : Finset ↑(setOf fun x' => Ne x' x)
        this : DecidableEq ι
        ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
        he : s'.Nonempty
        ha' : LT.lt s'.card (s'.biUnion fun x_1 => t ↑x_1).card
        hb : Not (Membership.mem (s'.biUnion fun x_1 => t ↑x_1) a)
        ⊢ LE.le s'.card (s'.biUnion fun x_1 => t ↑x_1).card
      -/
      exact Nat.le_of_lt ha'
      /-
        🎉 no goals
      -/
    /-
      case neg
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      x : ι
      a : α
      s' : Finset ↑(setOf fun x' => Ne x' x)
      this : DecidableEq ι
      ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
      he : Not s'.Nonempty
      ⊢ LE.le s'.card (s'.biUnion fun x' => (t ↑x').erase a).card
    -/
  · rw [nonempty_iff_ne_empty, not_not] at he
    /-
      case neg
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      x : ι
      a : α
      s' : Finset ↑(setOf fun x' => Ne x' x)
      this : DecidableEq ι
      ha : s'.Nonempty → Ne (Finset.image (fun z => ↑z) s') Finset.univ → LT.lt s'.c …
      he : Eq s' EmptyCollection.emptyCollection
      ⊢ LE.le s'.card (s'.biUnion fun x' => (t ↑x').erase a).card
    -/
    subst s'
    /-
      case neg
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      x : ι
      a : α
      this : DecidableEq ι
      ha : EmptyCollection.emptyCollection.Nonempty → Ne (Finset.image (fun z => ↑z) …
      ⊢ LE.le EmptyCollection.emptyCollection.card (EmptyCollection.emptyCollection. …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- First case of the inductive step: assuming that
`∀ (s : Finset ι), s.Nonempty → s ≠ univ → #s < #(s.biUnion t)`
and that the statement of **Hall's Marriage Theorem** is true for all
`ι'` of cardinality ≤ `n`, then it is true for `ι` of cardinality `n + 1`.
-/
theorem hall_hard_inductive_step_A {n : ℕ} (hn : Fintype.card ι = n + 1)
    (ht : ∀ s : Finset ι, #s ≤ #(s.biUnion t))
    (ih :
      ∀ {ι' : Type u} [Fintype ι'] (t' : ι' → Finset α),
        Fintype.card ι' ≤ n →
          (∀ s' : Finset ι', #s' ≤ #(s'.biUnion t')) →
            ∃ f : ι' → α, Function.Injective f ∧ ∀ x, f x ∈ t' x)
    (ha : ∀ s : Finset ι, s.Nonempty → s ≠ univ → #s < #(s.biUnion t)) :
    ∃ f : ι → α, Function.Injective f ∧ ∀ x, f x ∈ t x := by
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  haveI : Nonempty ι := Fintype.card_pos_iff.mp (hn.symm ▸ Nat.succ_pos _)
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    this : Nonempty ι
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  haveI := Classical.decEq ι
  -- Choose an arbitrary element `x : ι` and `y : t x`.
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    this✝ : Nonempty ι
    this : DecidableEq ι
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  let x := Classical.arbitrary ι
  have tx_ne : (t x).Nonempty := by
    rw [← Finset.card_pos]
    calc
      0 < 1 := Nat.one_pos
      _ ≤ #(.biUnion {x} t) := ht {x}
      _ = (t x).card := by rw [Finset.singleton_biUnion]

  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    this✝ : Nonempty ι
    this : DecidableEq ι
    x : ι := Classical.arbitrary ι
    tx_ne : (t x).Nonempty
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  choose y hy using tx_ne
  -- Restrict to everything except `x` and `y`.
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    this✝ : Nonempty ι
    this : DecidableEq ι
    x : ι := Classical.arbitrary ι
    y : α
    hy : Membership.mem (t x) y
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  let ι' := { x' : ι | x' ≠ x }
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    this✝ : Nonempty ι
    this : DecidableEq ι
    x : ι := Classical.arbitrary ι
    y : α
    hy : Membership.mem (t x) y
    ι' : Set ι := setOf fun x' => Ne x' x
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  let t' : ι' → Finset α := fun x' => (t x').erase y
  have card_ι' : Fintype.card ι' = n :=
    calc
      Fintype.card ι' = Fintype.card ι - 1 := Set.card_ne_eq _
      _ = n := by rw [hn, Nat.add_succ_sub_one, add_zero]

  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    this✝ : Nonempty ι
    this : DecidableEq ι
    x : ι := Classical.arbitrary ι
    y : α
    hy : Membership.mem (t x) y
    ι' : Set ι := setOf fun x' => Ne x' x
    t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
    card_ι' : Eq (Fintype.card ↑ι') n
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  rcases ih t' card_ι'.le (hall_cond_of_erase y ha) with ⟨f', hfinj, hfr⟩
  -- Extend the resulting function.
  /-
    case intro.intro
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
    this✝ : Nonempty ι
    this : DecidableEq ι
    x : ι := Classical.arbitrary ι
    y : α
    hy : Membership.mem (t x) y
    ι' : Set ι := setOf fun x' => Ne x' x
    t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
    card_ι' : Eq (Fintype.card ↑ι') n
    f' : ↑ι' → α
    hfinj : Function.Injective f'
    hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  refine ⟨fun z => if h : z = x then y else f' ⟨z, h⟩, ?_, ?_⟩
    /-
      case intro.intro.refine_1
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
      this✝ : Nonempty ι
      this : DecidableEq ι
      x : ι := Classical.arbitrary ι
      y : α
      hy : Membership.mem (t x) y
      ι' : Set ι := setOf fun x' => Ne x' x
      t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
      card_ι' : Eq (Fintype.card ↑ι') n
      f' : ↑ι' → α
      hfinj : Function.Injective f'
      hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
      ⊢ Function.Injective fun z => dite (Eq z x) (fun h => y) fun h => f' ⟨z, h⟩
    -/
  · rintro z₁ z₂
    have key : ∀ {x}, y ≠ f' x := by
      intro x h
      simpa [t', ← h] using hfr x
    /-
      case intro.intro.refine_1
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
      this✝ : Nonempty ι
      this : DecidableEq ι
      x : ι := Classical.arbitrary ι
      y : α
      hy : Membership.mem (t x) y
      ι' : Set ι := setOf fun x' => Ne x' x
      t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
      card_ι' : Eq (Fintype.card ↑ι') n
      f' : ↑ι' → α
      hfinj : Function.Injective f'
      hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
      z₁ z₂ : ι
      key : ∀ {x : ↑ι'}, Ne y (f' x)
      ⊢ Eq ((fun z => dite (Eq z x) (fun h => y) fun h => f' ⟨z, h⟩) z₁) ((fun z =>  …
    -/
    by_cases h₁ : z₁ = x <;> by_cases h₂ : z₂ = x <;>
      /-
        case pos
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        n : Nat
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
        ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
        this✝ : Nonempty ι
        this : DecidableEq ι
        x : ι := Classical.arbitrary ι
        y : α
        hy : Membership.mem (t x) y
        ι' : Set ι := setOf fun x' => Ne x' x
        t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
        card_ι' : Eq (Fintype.card ↑ι') n
        f' : ↑ι' → α
        hfinj : Function.Injective f'
        hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
        z₁ z₂ : ι
        key : ∀ {x : ↑ι'}, Ne y (f' x)
        h₁ : Eq z₁ x
        h₂ : Eq z₂ x
        ⊢ Eq ((fun z => dite (Eq z x) (fun h => y) fun h => f' ⟨z, h⟩) z₁) ((fun z =>  …
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
      simp [h₁, h₂, hfinj.eq_iff, key, key.symm]
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.refine_2
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
      this✝ : Nonempty ι
      this : DecidableEq ι
      x : ι := Classical.arbitrary ι
      y : α
      hy : Membership.mem (t x) y
      ι' : Set ι := setOf fun x' => Ne x' x
      t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
      card_ι' : Eq (Fintype.card ↑ι') n
      f' : ↑ι' → α
      hfinj : Function.Injective f'
      hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
      ⊢ ∀ (x_1 : ι), Membership.mem (t x_1) ((fun z => dite (Eq z x) (fun h => y) fu …
    -/
  · intro z
    /-
      case intro.intro.refine_2
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
      this✝ : Nonempty ι
      this : DecidableEq ι
      x : ι := Classical.arbitrary ι
      y : α
      hy : Membership.mem (t x) y
      ι' : Set ι := setOf fun x' => Ne x' x
      t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
      card_ι' : Eq (Fintype.card ↑ι') n
      f' : ↑ι' → α
      hfinj : Function.Injective f'
      hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
      z : ι
      ⊢ Membership.mem (t z) ((fun z => dite (Eq z x) (fun h => y) fun h => f' ⟨z, h …
    -/
    simp only [ne_eq, Set.mem_setOf_eq]
    /-
      case intro.intro.refine_2
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
      this✝ : Nonempty ι
      this : DecidableEq ι
      x : ι := Classical.arbitrary ι
      y : α
      hy : Membership.mem (t x) y
      ι' : Set ι := setOf fun x' => Ne x' x
      t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
      card_ι' : Eq (Fintype.card ↑ι') n
      f' : ↑ι' → α
      hfinj : Function.Injective f'
      hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
      z : ι
      ⊢ Membership.mem (t z) (dite (Eq z x) (fun h => y) fun h => f' ⟨z, h⟩)
    -/
    split_ifs with hz
      /-
        case pos
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        n : Nat
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
        ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
        this✝ : Nonempty ι
        this : DecidableEq ι
        x : ι := Classical.arbitrary ι
        y : α
        hy : Membership.mem (t x) y
        ι' : Set ι := setOf fun x' => Ne x' x
        t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
        card_ι' : Eq (Fintype.card ↑ι') n
        f' : ↑ι' → α
        hfinj : Function.Injective f'
        hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
        z : ι
        hz : Eq z x
        ⊢ Membership.mem (t z) y
      -/
    · rwa [hz]
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        n : Nat
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
        ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
        this✝ : Nonempty ι
        this : DecidableEq ι
        x : ι := Classical.arbitrary ι
        y : α
        hy : Membership.mem (t x) y
        ι' : Set ι := setOf fun x' => Ne x' x
        t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
        card_ι' : Eq (Fintype.card ↑ι') n
        f' : ↑ι' → α
        hfinj : Function.Injective f'
        hfr : ∀ (x : ↑ι'), Membership.mem (t' x) (f' x)
        z : ι
        hz : Not (Eq z x)
        ⊢ Membership.mem (t z) (f' ⟨z, hz⟩)
      -/
    · specialize hfr ⟨z, hz⟩
      /-
        case neg
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        n : Nat
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
        ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
        this✝ : Nonempty ι
        this : DecidableEq ι
        x : ι := Classical.arbitrary ι
        y : α
        hy : Membership.mem (t x) y
        ι' : Set ι := setOf fun x' => Ne x' x
        t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
        card_ι' : Eq (Fintype.card ↑ι') n
        f' : ↑ι' → α
        hfinj : Function.Injective f'
        z : ι
        hz : Not (Eq z x)
        hfr : Membership.mem (t' ⟨z, hz⟩) (f' ⟨z, hz⟩)
        ⊢ Membership.mem (t z) (f' ⟨z, hz⟩)
      -/
      rw [mem_erase] at hfr
      /-
        case neg
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        n : Nat
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
        ha : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion …
        this✝ : Nonempty ι
        this : DecidableEq ι
        x : ι := Classical.arbitrary ι
        y : α
        hy : Membership.mem (t x) y
        ι' : Set ι := setOf fun x' => Ne x' x
        t' : ↑ι' → Finset α := fun x' => (t ↑x').erase y
        card_ι' : Eq (Fintype.card ↑ι') n
        f' : ↑ι' → α
        hfinj : Function.Injective f'
        z : ι
        hz : Not (Eq z x)
        hfr : And (Ne (f' ⟨z, hz⟩) y) (Membership.mem (t ↑⟨z, hz⟩) (f' ⟨z, hz⟩))
        ⊢ Membership.mem (t z) (f' ⟨z, hz⟩)
      -/
      exact hfr.2
      /-
        🎉 no goals
      -/


theorem hall_cond_of_restrict {ι : Type u} {t : ι → Finset α} {s : Finset ι}
    (ht : ∀ s : Finset ι, #s ≤ #(s.biUnion t)) (s' : Finset (s : Set ι)) :
    #s' ≤ #(s'.biUnion fun a' => t a') := by
  classical
    rw [← card_image_of_injective s' Subtype.coe_injective]
    convert ht (s'.image fun z => z.1) using 1
    apply congr_arg
    ext y
    simp


theorem hall_cond_of_compl {ι : Type u} {t : ι → Finset α} {s : Finset ι}
    (hus : #s = #(s.biUnion t)) (ht : ∀ s : Finset ι, #s ≤ #(s.biUnion t))
    (s' : Finset (sᶜ : Set ι)) : #s' ≤ #(s'.biUnion fun x' => t x' \ s.biUnion t) := by
  /-
    α : Type v
    inst✝ : DecidableEq α
    ι : Type u
    t : ι → Finset α
    s : Finset ι
    hus : Eq s.card (s.biUnion t).card
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    s' : Finset ↑(HasCompl.compl ↑s)
    ⊢ LE.le s'.card (s'.biUnion fun x' => SDiff.sdiff (t ↑x') (s.biUnion t)).card
  -/
  haveI := Classical.decEq ι
  have disj : Disjoint s (s'.image fun z => z.1) := by
    simp only [disjoint_left, not_exists, mem_image, exists_prop, SetCoe.exists, exists_and_right,
      exists_eq_right, Subtype.coe_mk]
    intro x hx hc _
    exact absurd hx hc
  have : #s' = #(s ∪ s'.image fun z => z.1) - #s := by
    simp [disj, card_image_of_injective _ Subtype.coe_injective, Nat.add_sub_cancel_left]
  /-
    α : Type v
    inst✝ : DecidableEq α
    ι : Type u
    t : ι → Finset α
    s : Finset ι
    hus : Eq s.card (s.biUnion t).card
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    s' : Finset ↑(HasCompl.compl ↑s)
    this✝ : DecidableEq ι
    disj : Disjoint s (Finset.image (fun z => ↑z) s')
    this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
    ⊢ LE.le s'.card (s'.biUnion fun x' => SDiff.sdiff (t ↑x') (s.biUnion t)).card
  -/
  rw [this, hus]
  /-
    α : Type v
    inst✝ : DecidableEq α
    ι : Type u
    t : ι → Finset α
    s : Finset ι
    hus : Eq s.card (s.biUnion t).card
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    s' : Finset ↑(HasCompl.compl ↑s)
    this✝ : DecidableEq ι
    disj : Disjoint s (Finset.image (fun z => ↑z) s')
    this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
    ⊢ LE.le (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).card (s.biU …
  -/
  refine (Nat.sub_le_sub_right (ht _) _).trans ?_
  /-
    α : Type v
    inst✝ : DecidableEq α
    ι : Type u
    t : ι → Finset α
    s : Finset ι
    hus : Eq s.card (s.biUnion t).card
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    s' : Finset ↑(HasCompl.compl ↑s)
    this✝ : DecidableEq ι
    disj : Disjoint s (Finset.image (fun z => ↑z) s')
    this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
    ⊢ LE.le (HSub.hSub ((Union.union s (Finset.image (fun z => ↑z) s')).biUnion t) …
  -/
  rw [← card_sdiff]
    /-
      α : Type v
      inst✝ : DecidableEq α
      ι : Type u
      t : ι → Finset α
      s : Finset ι
      hus : Eq s.card (s.biUnion t).card
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      s' : Finset ↑(HasCompl.compl ↑s)
      this✝ : DecidableEq ι
      disj : Disjoint s (Finset.image (fun z => ↑z) s')
      this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
      ⊢ LE.le (SDiff.sdiff ((Union.union s (Finset.image (fun z => ↑z) s')).biUnion  …
    -/
  · refine (card_le_card ?_).trans le_rfl
    /-
      α : Type v
      inst✝ : DecidableEq α
      ι : Type u
      t : ι → Finset α
      s : Finset ι
      hus : Eq s.card (s.biUnion t).card
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      s' : Finset ↑(HasCompl.compl ↑s)
      this✝ : DecidableEq ι
      disj : Disjoint s (Finset.image (fun z => ↑z) s')
      this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
      ⊢ HasSubset.Subset (SDiff.sdiff ((Union.union s (Finset.image (fun z => ↑z) s' …
    -/
    intro t
    simp only [mem_biUnion, mem_sdiff, not_exists, mem_image, and_imp, mem_union, exists_and_right,
      exists_imp]
    /-
      α : Type v
      inst✝ : DecidableEq α
      ι : Type u
      t✝ : ι → Finset α
      s : Finset ι
      hus : Eq s.card (s.biUnion t✝).card
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t✝).card
      s' : Finset ↑(HasCompl.compl ↑s)
      this✝ : DecidableEq ι
      disj : Disjoint s (Finset.image (fun z => ↑z) s')
      this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
      t : α
      ⊢ ∀ (x : ι), Or (Membership.mem s x) (Exists fun a => And (Membership.mem s' a …
    -/
    rintro x (hx | ⟨x', hx', rfl⟩) rat hs
      /-
        case inl
        α : Type v
        inst✝ : DecidableEq α
        ι : Type u
        t✝ : ι → Finset α
        s : Finset ι
        hus : Eq s.card (s.biUnion t✝).card
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t✝).card
        s' : Finset ↑(HasCompl.compl ↑s)
        this✝ : DecidableEq ι
        disj : Disjoint s (Finset.image (fun z => ↑z) s')
        this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
        t : α
        x : ι
        hx : Membership.mem s x
        rat : Membership.mem (t✝ x) t
        hs : ∀ (x : ι), Not (And (Membership.mem s x) (Membership.mem (t✝ x) t))
        ⊢ Exists fun a => And (Membership.mem s' a) (And (Membership.mem (t✝ ↑a) t) (∀ …
      -/
    · exact False.elim <| (hs x) <| And.intro hx rat
      /-
        🎉 no goals
      -/
      /-
        case inr.intro.intro
        α : Type v
        inst✝ : DecidableEq α
        ι : Type u
        t✝ : ι → Finset α
        s : Finset ι
        hus : Eq s.card (s.biUnion t✝).card
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t✝).card
        s' : Finset ↑(HasCompl.compl ↑s)
        this✝ : DecidableEq ι
        disj : Disjoint s (Finset.image (fun z => ↑z) s')
        this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
        t : α
        x' : ↑(HasCompl.compl ↑s)
        hx' : Membership.mem s' x'
        rat : Membership.mem (t✝ ↑x') t
        hs : ∀ (x : ι), Not (And (Membership.mem s x) (Membership.mem (t✝ x) t))
        ⊢ Exists fun a => And (Membership.mem s' a) (And (Membership.mem (t✝ ↑a) t) (∀ …
      -/
    · use x', hx', rat, hs
      /-
        🎉 no goals
      -/
    /-
      α : Type v
      inst✝ : DecidableEq α
      ι : Type u
      t : ι → Finset α
      s : Finset ι
      hus : Eq s.card (s.biUnion t).card
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      s' : Finset ↑(HasCompl.compl ↑s)
      this✝ : DecidableEq ι
      disj : Disjoint s (Finset.image (fun z => ↑z) s')
      this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
      ⊢ HasSubset.Subset (s.biUnion t) ((Union.union s (Finset.image (fun z => ↑z) s …
    -/
  · apply biUnion_subset_biUnion_of_subset_left
    /-
      case h
      α : Type v
      inst✝ : DecidableEq α
      ι : Type u
      t : ι → Finset α
      s : Finset ι
      hus : Eq s.card (s.biUnion t).card
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      s' : Finset ↑(HasCompl.compl ↑s)
      this✝ : DecidableEq ι
      disj : Disjoint s (Finset.image (fun z => ↑z) s')
      this : Eq s'.card (HSub.hSub (Union.union s (Finset.image (fun z => ↑z) s')).c …
      ⊢ HasSubset.Subset s (Union.union s (Finset.image (fun z => ↑z) s'))
    -/
    apply subset_union_left
    /-
      🎉 no goals
    -/


/-- Second case of the inductive step: assuming that
`∃ (s : Finset ι), s ≠ univ → #s = #(s.biUnion t)`
and that the statement of **Hall's Marriage Theorem** is true for all
`ι'` of cardinality ≤ `n`, then it is true for `ι` of cardinality `n + 1`.
-/
theorem hall_hard_inductive_step_B {n : ℕ} (hn : Fintype.card ι = n + 1)
    (ht : ∀ s : Finset ι, #s ≤ #(s.biUnion t))
    (ih :
      ∀ {ι' : Type u} [Fintype ι'] (t' : ι' → Finset α),
        Fintype.card ι' ≤ n →
          (∀ s' : Finset ι', #s' ≤ #(s'.biUnion t')) →
            ∃ f : ι' → α, Function.Injective f ∧ ∀ x, f x ∈ t' x)
    (s : Finset ι) (hs : s.Nonempty) (hns : s ≠ univ) (hus : #s = #(s.biUnion t)) :
    ∃ f : ι → α, Function.Injective f ∧ ∀ x, f x ∈ t x := by
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  haveI := Classical.decEq ι
  -- Restrict to `s`
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    this : DecidableEq ι
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  rw [Nat.add_one] at hn
  have card_ι'_le : Fintype.card s ≤ n := by
    apply Nat.le_of_lt_succ
    calc
      Fintype.card s = #s := Fintype.card_coe _
      _ < Fintype.card ι := (card_lt_iff_ne_univ _).mpr hns
      _ = n.succ := hn
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) n.succ
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    this : DecidableEq ι
    card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  let t' : s → Finset α := fun x' => t x'
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) n.succ
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    this : DecidableEq ι
    card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
    t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  rcases ih t' card_ι'_le (hall_cond_of_restrict ht) with ⟨f', hf', hsf'⟩
  -- Restrict to `sᶜ` in the domain and `(s.biUnion t)ᶜ` in the codomain.
  /-
    case intro.intro
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) n.succ
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    this : DecidableEq ι
    card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
    t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
    f' : (Subtype fun x => Membership.mem s x) → α
    hf' : Function.Injective f'
    hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  set ι'' := (s : Set ι)ᶜ
  /-
    case intro.intro
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) n.succ
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    this : DecidableEq ι
    card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
    t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
    f' : (Subtype fun x => Membership.mem s x) → α
    hf' : Function.Injective f'
    hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
    ι'' : Set ι := HasCompl.compl ↑s
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  let t'' : ι'' → Finset α := fun a'' => t a'' \ s.biUnion t
  have card_ι''_le : Fintype.card ι'' ≤ n := by
    simp_rw [ι'', ← Nat.lt_succ_iff, ← hn, ← Finset.coe_compl, coe_sort_coe]
    rwa [Fintype.card_coe, card_compl_lt_iff_nonempty]
  /-
    case intro.intro
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) n.succ
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    this : DecidableEq ι
    card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
    t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
    f' : (Subtype fun x => Membership.mem s x) → α
    hf' : Function.Injective f'
    hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
    ι'' : Set ι := HasCompl.compl ↑s
    t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
    card_ι''_le : LE.le (Fintype.card ↑ι'') n
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  rcases ih t'' card_ι''_le (hall_cond_of_compl hus ht) with ⟨f'', hf'', hsf''⟩
  -- Put them together
  have f'_mem_biUnion : ∀ (x') (hx' : x' ∈ s), f' ⟨x', hx'⟩ ∈ s.biUnion t := by
    intro x' hx'
    rw [mem_biUnion]
    exact ⟨x', hx', hsf' _⟩
  have f''_not_mem_biUnion : ∀ (x'') (hx'' : ¬x'' ∈ s), ¬f'' ⟨x'', hx''⟩ ∈ s.biUnion t := by
    intro x'' hx''
    have h := hsf'' ⟨x'', hx''⟩
    rw [mem_sdiff] at h
    exact h.2
  have im_disj :
      ∀ (x' x'' : ι) (hx' : x' ∈ s) (hx'' : ¬x'' ∈ s), f' ⟨x', hx'⟩ ≠ f'' ⟨x'', hx''⟩ := by
    intro x x' hx' hx'' h
    apply f''_not_mem_biUnion x' hx''
    rw [← h]
    apply f'_mem_biUnion x
  /-
    case intro.intro.intro.intro
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Fintype ι
    n : Nat
    hn : Eq (Fintype.card ι) n.succ
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
    s : Finset ι
    hs : s.Nonempty
    hns : Ne s Finset.univ
    hus : Eq s.card (s.biUnion t).card
    this : DecidableEq ι
    card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
    t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
    f' : (Subtype fun x => Membership.mem s x) → α
    hf' : Function.Injective f'
    hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
    ι'' : Set ι := HasCompl.compl ↑s
    t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
    card_ι''_le : LE.le (Fintype.card ↑ι'') n
    f'' : ↑ι'' → α
    hf'' : Function.Injective f''
    hsf'' : ∀ (x : ↑ι''), Membership.mem (t'' x) (f'' x)
    f'_mem_biUnion : ∀ (x' : ι) (hx' : Membership.mem s x'), Membership.mem (s.biU …
    f''_not_mem_biUnion : ∀ (x'' : ι) (hx'' : Not (Membership.mem s x'')), Not (Me …
    im_disj : ∀ (x' x'' : ι) (hx' : Membership.mem s x') (hx'' : Not (Membership.m …
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  refine ⟨fun x => if h : x ∈ s then f' ⟨x, h⟩ else f'' ⟨x, h⟩, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) n.succ
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      s : Finset ι
      hs : s.Nonempty
      hns : Ne s Finset.univ
      hus : Eq s.card (s.biUnion t).card
      this : DecidableEq ι
      card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
      t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
      f' : (Subtype fun x => Membership.mem s x) → α
      hf' : Function.Injective f'
      hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
      ι'' : Set ι := HasCompl.compl ↑s
      t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
      card_ι''_le : LE.le (Fintype.card ↑ι'') n
      f'' : ↑ι'' → α
      hf'' : Function.Injective f''
      hsf'' : ∀ (x : ↑ι''), Membership.mem (t'' x) (f'' x)
      f'_mem_biUnion : ∀ (x' : ι) (hx' : Membership.mem s x'), Membership.mem (s.biU …
      f''_not_mem_biUnion : ∀ (x'' : ι) (hx'' : Not (Membership.mem s x'')), Not (Me …
      im_disj : ∀ (x' x'' : ι) (hx' : Membership.mem s x') (hx'' : Not (Membership.m …
      ⊢ Function.Injective fun x => dite (Membership.mem s x) (fun h => f' ⟨x, h⟩) f …
    -/
  · refine hf'.dite _ hf'' (@fun x x' => im_disj x x' _ _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) n.succ
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      s : Finset ι
      hs : s.Nonempty
      hns : Ne s Finset.univ
      hus : Eq s.card (s.biUnion t).card
      this : DecidableEq ι
      card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
      t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
      f' : (Subtype fun x => Membership.mem s x) → α
      hf' : Function.Injective f'
      hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
      ι'' : Set ι := HasCompl.compl ↑s
      t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
      card_ι''_le : LE.le (Fintype.card ↑ι'') n
      f'' : ↑ι'' → α
      hf'' : Function.Injective f''
      hsf'' : ∀ (x : ↑ι''), Membership.mem (t'' x) (f'' x)
      f'_mem_biUnion : ∀ (x' : ι) (hx' : Membership.mem s x'), Membership.mem (s.biU …
      f''_not_mem_biUnion : ∀ (x'' : ι) (hx'' : Not (Membership.mem s x'')), Not (Me …
      im_disj : ∀ (x' x'' : ι) (hx' : Membership.mem s x') (hx'' : Not (Membership.m …
      ⊢ ∀ (x : ι), Membership.mem (t x) ((fun x => dite (Membership.mem s x) (fun h  …
    -/
  · intro x
    /-
      case intro.intro.intro.intro.refine_2
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) n.succ
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      s : Finset ι
      hs : s.Nonempty
      hns : Ne s Finset.univ
      hus : Eq s.card (s.biUnion t).card
      this : DecidableEq ι
      card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
      t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
      f' : (Subtype fun x => Membership.mem s x) → α
      hf' : Function.Injective f'
      hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
      ι'' : Set ι := HasCompl.compl ↑s
      t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
      card_ι''_le : LE.le (Fintype.card ↑ι'') n
      f'' : ↑ι'' → α
      hf'' : Function.Injective f''
      hsf'' : ∀ (x : ↑ι''), Membership.mem (t'' x) (f'' x)
      f'_mem_biUnion : ∀ (x' : ι) (hx' : Membership.mem s x'), Membership.mem (s.biU …
      f''_not_mem_biUnion : ∀ (x'' : ι) (hx'' : Not (Membership.mem s x'')), Not (Me …
      im_disj : ∀ (x' x'' : ι) (hx' : Membership.mem s x') (hx'' : Not (Membership.m …
      x : ι
      ⊢ Membership.mem (t x) ((fun x => dite (Membership.mem s x) (fun h => f' ⟨x, h …
    -/
    simp only [of_eq_true]
    /-
      case intro.intro.intro.intro.refine_2
      ι : Type u
      α : Type v
      inst✝¹ : DecidableEq α
      t : ι → Finset α
      inst✝ : Fintype ι
      n : Nat
      hn : Eq (Fintype.card ι) n.succ
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
      s : Finset ι
      hs : s.Nonempty
      hns : Ne s Finset.univ
      hus : Eq s.card (s.biUnion t).card
      this : DecidableEq ι
      card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
      t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
      f' : (Subtype fun x => Membership.mem s x) → α
      hf' : Function.Injective f'
      hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
      ι'' : Set ι := HasCompl.compl ↑s
      t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
      card_ι''_le : LE.le (Fintype.card ↑ι'') n
      f'' : ↑ι'' → α
      hf'' : Function.Injective f''
      hsf'' : ∀ (x : ↑ι''), Membership.mem (t'' x) (f'' x)
      f'_mem_biUnion : ∀ (x' : ι) (hx' : Membership.mem s x'), Membership.mem (s.biU …
      f''_not_mem_biUnion : ∀ (x'' : ι) (hx'' : Not (Membership.mem s x'')), Not (Me …
      im_disj : ∀ (x' x'' : ι) (hx' : Membership.mem s x') (hx'' : Not (Membership.m …
      x : ι
      ⊢ Membership.mem (t x) (dite (Membership.mem s x) (fun h => f' ⟨x, h⟩) fun h = …
    -/
    split_ifs with h
      /-
        case pos
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        n : Nat
        hn : Eq (Fintype.card ι) n.succ
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
        s : Finset ι
        hs : s.Nonempty
        hns : Ne s Finset.univ
        hus : Eq s.card (s.biUnion t).card
        this : DecidableEq ι
        card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
        t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
        f' : (Subtype fun x => Membership.mem s x) → α
        hf' : Function.Injective f'
        hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
        ι'' : Set ι := HasCompl.compl ↑s
        t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
        card_ι''_le : LE.le (Fintype.card ↑ι'') n
        f'' : ↑ι'' → α
        hf'' : Function.Injective f''
        hsf'' : ∀ (x : ↑ι''), Membership.mem (t'' x) (f'' x)
        f'_mem_biUnion : ∀ (x' : ι) (hx' : Membership.mem s x'), Membership.mem (s.biU …
        f''_not_mem_biUnion : ∀ (x'' : ι) (hx'' : Not (Membership.mem s x'')), Not (Me …
        im_disj : ∀ (x' x'' : ι) (hx' : Membership.mem s x') (hx'' : Not (Membership.m …
        x : ι
        h : Membership.mem s x
        ⊢ Membership.mem (t x) (f' ⟨x, h⟩)
      -/
    · exact hsf' ⟨x, h⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u
        α : Type v
        inst✝¹ : DecidableEq α
        t : ι → Finset α
        inst✝ : Fintype ι
        n : Nat
        hn : Eq (Fintype.card ι) n.succ
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        ih : ∀ {ι' : Type u} [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype. …
        s : Finset ι
        hs : s.Nonempty
        hns : Ne s Finset.univ
        hus : Eq s.card (s.biUnion t).card
        this : DecidableEq ι
        card_ι'_le : LE.le (Fintype.card (Subtype fun x => Membership.mem s x)) n
        t' : (Subtype fun x => Membership.mem s x) → Finset α := fun x' => t ↑x'
        f' : (Subtype fun x => Membership.mem s x) → α
        hf' : Function.Injective f'
        hsf' : ∀ (x : Subtype fun x => Membership.mem s x), Membership.mem (t' x) (f' x)
        ι'' : Set ι := HasCompl.compl ↑s
        t'' : ↑ι'' → Finset α := fun a'' => SDiff.sdiff (t ↑a'') (s.biUnion t)
        card_ι''_le : LE.le (Fintype.card ↑ι'') n
        f'' : ↑ι'' → α
        hf'' : Function.Injective f''
        hsf'' : ∀ (x : ↑ι''), Membership.mem (t'' x) (f'' x)
        f'_mem_biUnion : ∀ (x' : ι) (hx' : Membership.mem s x'), Membership.mem (s.biU …
        f''_not_mem_biUnion : ∀ (x'' : ι) (hx'' : Not (Membership.mem s x'')), Not (Me …
        im_disj : ∀ (x' x'' : ι) (hx' : Membership.mem s x') (hx'' : Not (Membership.m …
        x : ι
        h : Not (Membership.mem s x)
        ⊢ Membership.mem (t x) (f'' ⟨x, h⟩)
      -/
    · exact sdiff_subset (hsf'' ⟨x, h⟩)
      /-
        🎉 no goals
      -/


/-- Here we combine the two inductive steps into a full strong induction proof,
completing the proof the harder direction of **Hall's Marriage Theorem**.
-/
theorem hall_hard_inductive (ht : ∀ s : Finset ι, #s ≤ #(s.biUnion t)) :
    ∃ f : ι → α, Function.Injective f ∧ ∀ x, f x ∈ t x := by
  /-
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Finite ι
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u
    α : Type v
    inst✝¹ : DecidableEq α
    t : ι → Finset α
    inst✝ : Finite ι
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    val✝ : Fintype ι
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  induction' hn : Fintype.card ι using Nat.strong_induction_on with n ih generalizing ι
  /-
    case intro.h
    α : Type v
    inst✝¹ : DecidableEq α
    n : Nat
    ih : ∀ (m : Nat), LT.lt m n → ∀ {ι : Type u} {t : ι → Finset α} [inst : Finite …
    ι : Type u
    t : ι → Finset α
    inst✝ : Finite ι
    ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
    val✝ : Fintype ι
    hn : Eq (Fintype.card ι) n
    ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
  -/
  rcases n with (_ | n)
    /-
      case intro.h.zero
      α : Type v
      inst✝¹ : DecidableEq α
      ι : Type u
      t : ι → Finset α
      inst✝ : Finite ι
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      val✝ : Fintype ι
      ih : ∀ (m : Nat), LT.lt m 0 → ∀ {ι : Type u} {t : ι → Finset α} [inst : Finite …
      hn : Eq (Fintype.card ι) 0
      ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
    -/
  · rw [Fintype.card_eq_zero_iff] at hn
    /-
      case intro.h.zero
      α : Type v
      inst✝¹ : DecidableEq α
      ι : Type u
      t : ι → Finset α
      inst✝ : Finite ι
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      val✝ : Fintype ι
      ih : ∀ (m : Nat), LT.lt m 0 → ∀ {ι : Type u} {t : ι → Finset α} [inst : Finite …
      hn : IsEmpty ι
      ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
    -/
    exact ⟨isEmptyElim, isEmptyElim, isEmptyElim⟩
    /-
      🎉 no goals
    -/
  · have ih' : ∀ (ι' : Type u) [Fintype ι'] (t' : ι' → Finset α), Fintype.card ι' ≤ n →
        (∀ s' : Finset ι', #s' ≤ #(s'.biUnion t')) →
        ∃ f : ι' → α, Function.Injective f ∧ ∀ x, f x ∈ t' x := by
      intro ι' _ _ hι' ht'
      exact ih _ (Nat.lt_succ_of_le hι') ht' _ rfl
    /-
      case intro.h.succ
      α : Type v
      inst✝¹ : DecidableEq α
      ι : Type u
      t : ι → Finset α
      inst✝ : Finite ι
      ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
      val✝ : Fintype ι
      n : Nat
      ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {ι : Type u} {t : ι → Finset α}  …
      hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
      ih' : ∀ (ι' : Type u) [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype …
      ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
    -/
    by_cases h : ∀ s : Finset ι, s.Nonempty → s ≠ univ → #s < #(s.biUnion t)
      /-
        case pos
        α : Type v
        inst✝¹ : DecidableEq α
        ι : Type u
        t : ι → Finset α
        inst✝ : Finite ι
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        val✝ : Fintype ι
        n : Nat
        ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {ι : Type u} {t : ι → Finset α}  …
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ih' : ∀ (ι' : Type u) [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype …
        h : ∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biUnion  …
        ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
      -/
    · refine hall_hard_inductive_step_A hn ht (@fun ι' => ih' ι') h
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type v
        inst✝¹ : DecidableEq α
        ι : Type u
        t : ι → Finset α
        inst✝ : Finite ι
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        val✝ : Fintype ι
        n : Nat
        ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {ι : Type u} {t : ι → Finset α}  …
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ih' : ∀ (ι' : Type u) [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype …
        h : Not (∀ (s : Finset ι), s.Nonempty → Ne s Finset.univ → LT.lt s.card (s.biU …
        ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
      -/
    · push_neg at h
      /-
        case neg
        α : Type v
        inst✝¹ : DecidableEq α
        ι : Type u
        t : ι → Finset α
        inst✝ : Finite ι
        ht : ∀ (s : Finset ι), LE.le s.card (s.biUnion t).card
        val✝ : Fintype ι
        n : Nat
        ih : ∀ (m : Nat), LT.lt m (HAdd.hAdd n 1) → ∀ {ι : Type u} {t : ι → Finset α}  …
        hn : Eq (Fintype.card ι) (HAdd.hAdd n 1)
        ih' : ∀ (ι' : Type u) [inst : Fintype ι'] (t' : ι' → Finset α), LE.le (Fintype …
        h : Exists fun s => And s.Nonempty (And (Ne s Finset.univ) (LE.le (s.biUnion t …
        ⊢ Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x)  …
      -/
      rcases h with ⟨s, sne, snu, sle⟩
      exact hall_hard_inductive_step_B hn ht (@fun ι' => ih' ι')
        s sne snu (Nat.le_antisymm (ht _) sle)


/-- This is the version of **Hall's Marriage Theorem** in terms of indexed
families of finite sets `t : ι → Finset α` with `ι` finite.
It states that there is a set of distinct representatives if and only
if every union of `k` of the sets has at least `k` elements.

See `Finset.all_card_le_biUnion_card_iff_exists_injective` for a version
where the `Finite ι` constraint is removed.
-/
theorem Finset.all_card_le_biUnion_card_iff_existsInjective' {ι α : Type*} [Finite ι]
    [DecidableEq α] (t : ι → Finset α) :
    (∀ s : Finset ι, #s ≤ #(s.biUnion t)) ↔
      ∃ f : ι → α, Function.Injective f ∧ ∀ x, f x ∈ t x := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : Finite ι
    inst✝ : DecidableEq α
    t : ι → Finset α
    ⊢ Iff (∀ (s : Finset ι), LE.le s.card (s.biUnion t).card) (Exists fun f => And …
  -/
  constructor
    /-
      case mp
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      ⊢ (∀ (s : Finset ι), LE.le s.card (s.biUnion t).card) → Exists fun f => And (F …
    -/
  · exact HallMarriageTheorem.hall_hard_inductive
    /-
      🎉 no goals
    -/
    /-
      case mpr
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      ⊢ (Exists fun f => And (Function.Injective f) (∀ (x : ι), Membership.mem (t x) …
    -/
  · rintro ⟨f, hf₁, hf₂⟩ s
    /-
      case mpr.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      ⊢ LE.le s.card (s.biUnion t).card
    -/
    rw [← card_image_of_injective s hf₁]
    /-
      case mpr.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      ⊢ LE.le (Finset.image f s).card (s.biUnion t).card
    -/
    apply card_le_card
    /-
      case mpr.intro.intro.a
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      ⊢ HasSubset.Subset (Finset.image f s) (s.biUnion t)
    -/
    intro
    /-
      case mpr.intro.intro.a
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      a✝ : α
      ⊢ Membership.mem (Finset.image f s) a✝ → Membership.mem (s.biUnion t) a✝
    -/
    rw [mem_image, mem_biUnion]
    /-
      case mpr.intro.intro.a
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      a✝ : α
      ⊢ (Exists fun a => And (Membership.mem s a) (Eq (f a) a✝)) → Exists fun a => A …
    -/
    rintro ⟨x, hx, rfl⟩
    /-
      case mpr.intro.intro.a.intro.intro
      ι : Type u_1
      α : Type u_2
      inst✝¹ : Finite ι
      inst✝ : DecidableEq α
      t : ι → Finset α
      f : ι → α
      hf₁ : Function.Injective f
      hf₂ : ∀ (x : ι), Membership.mem (t x) (f x)
      s : Finset ι
      x : ι
      hx : Membership.mem s x
      ⊢ Exists fun a => And (Membership.mem s a) (Membership.mem (t a) (f x))
    -/
    exact ⟨x, hx, hf₂ x⟩
    /-
      🎉 no goals
    -/

