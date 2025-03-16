/-- Let `G` be a Type with multiplication, let `A B : Finset G` be finite subsets and
let `a0 b0 : G` be two elements.  `UniqueMul A B a0 b0` asserts `a0 * b0` can be written in at
most one way as a product of an element of `A` and an element of `B`. -/
@[to_additive
      "Let `G` be a Type with addition, let `A B : Finset G` be finite subsets and
let `a0 b0 : G` be two elements.  `UniqueAdd A B a0 b0` asserts `a0 + b0` can be written in at
most one way as a sum of an element from `A` and an element from `B`."]
def UniqueMul {G} [Mul G] (A B : Finset G) (a0 b0 : G) : Prop :=
  ∀ ⦃a b⦄, a ∈ A → b ∈ B → a * b = a0 * b0 → a = a0 ∧ b = b0


@[to_additive (attr := nontriviality, simp)]
theorem of_subsingleton [Subsingleton G] : UniqueMul A B a0 b0 := by
  /-
    G : Type u_1
    inst✝¹ : Mul G
    A B : Finset G
    a0 b0 : G
    inst✝ : Subsingleton G
    ⊢ UniqueMul A B a0 b0
  -/
  simp [UniqueMul, eq_iff_true_of_subsingleton]
  /-
    🎉 no goals
  -/


@[to_additive of_card_le_one]
theorem of_card_le_one (hA : A.Nonempty) (hB : B.Nonempty) (hA1 : #A ≤ 1) (hB1 : #B ≤ 1) :
    ∃ a ∈ A, ∃ b ∈ B, UniqueMul A B a b := by
  /-
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    hA : A.Nonempty
    hB : B.Nonempty
    hA1 : LE.le A.card 1
    hB1 : LE.le B.card 1
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  rw [Finset.card_le_one_iff] at hA1 hB1
  /-
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    hA : A.Nonempty
    hB : B.Nonempty
    hA1 : ∀ {a b : G}, Membership.mem A a → Membership.mem A b → Eq a b
    hB1 : ∀ {a b : G}, Membership.mem B a → Membership.mem B b → Eq a b
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  obtain ⟨a, ha⟩ := hA; obtain ⟨b, hb⟩ := hB
  /-
    case intro.intro
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    hA1 : ∀ {a b : G}, Membership.mem A a → Membership.mem A b → Eq a b
    hB1 : ∀ {a b : G}, Membership.mem B a → Membership.mem B b → Eq a b
    a : G
    ha : Membership.mem A a
    b : G
    hb : Membership.mem B b
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  exact ⟨a, ha, b, hb, fun _ _ ha' hb' _ ↦ ⟨hA1 ha' ha, hB1 hb' hb⟩⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-23")]
alias _root_.UniqueAdd.of_card_nonpos := UniqueAdd.of_card_le_one


@[to_additive]
theorem mt (h : UniqueMul A B a0 b0) :
    ∀ ⦃a b⦄, a ∈ A → b ∈ B → a ≠ a0 ∨ b ≠ b0 → a * b ≠ a0 * b0 := fun _ _ ha hb k ↦ by
  /-
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    a0 b0 : G
    h : UniqueMul A B a0 b0
    x✝¹ x✝ : G
    ha : Membership.mem A x✝¹
    hb : Membership.mem B x✝
    k : Or (Ne x✝¹ a0) (Ne x✝ b0)
    ⊢ Ne (HMul.hMul x✝¹ x✝) (HMul.hMul a0 b0)
  -/
  contrapose! k
  /-
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    a0 b0 : G
    h : UniqueMul A B a0 b0
    x✝¹ x✝ : G
    ha : Membership.mem A x✝¹
    hb : Membership.mem B x✝
    k : Eq (HMul.hMul x✝¹ x✝) (HMul.hMul a0 b0)
    ⊢ And (Eq x✝¹ a0) (Eq x✝ b0)
  -/
  exact h ha hb k
  /-
    🎉 no goals
  -/


@[to_additive]
theorem subsingleton (h : UniqueMul A B a0 b0) :
    Subsingleton { ab : G × G // ab.1 ∈ A ∧ ab.2 ∈ B ∧ ab.1 * ab.2 = a0 * b0 } :=
  ⟨fun ⟨⟨_a, _b⟩, ha, hb, ab⟩ ⟨⟨_a', _b'⟩, ha', hb', ab'⟩ ↦
    Subtype.ext <|
      Prod.ext ((h ha hb ab).1.trans (h ha' hb' ab').1.symm) <|
        (h ha hb ab).2.trans (h ha' hb' ab').2.symm⟩


@[to_additive]
theorem set_subsingleton (h : UniqueMul A B a0 b0) :
    Set.Subsingleton { ab : G × G | ab.1 ∈ A ∧ ab.2 ∈ B ∧ ab.1 * ab.2 = a0 * b0 } := by
  rintro ⟨x1, y1⟩ (hx : x1 ∈ A ∧ y1 ∈ B ∧ x1 * y1 = a0 * b0) ⟨x2, y2⟩
    (hy : x2 ∈ A ∧ y2 ∈ B ∧ x2 * y2 = a0 * b0)
  /-
    case mk.mk
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    a0 b0 : G
    h : UniqueMul A B a0 b0
    x1 y1 : G
    hx : And (Membership.mem A x1) (And (Membership.mem B y1) (Eq (HMul.hMul x1 y1 …
    x2 y2 : G
    hy : And (Membership.mem A x2) (And (Membership.mem B y2) (Eq (HMul.hMul x2 y2 …
    ⊢ Eq { fst := x1, snd := y1 } { fst := x2, snd := y2 }
  -/
  rcases h hx.1 hx.2.1 hx.2.2 with ⟨rfl, rfl⟩
  /-
    case mk.mk.intro
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    x1 y1 x2 y2 : G
    h : UniqueMul A B x1 y1
    hx : And (Membership.mem A x1) (And (Membership.mem B y1) (Eq (HMul.hMul x1 y1 …
    hy : And (Membership.mem A x2) (And (Membership.mem B y2) (Eq (HMul.hMul x2 y2 …
    ⊢ Eq { fst := x1, snd := y1 } { fst := x2, snd := y2 }
  -/
  rcases h hy.1 hy.2.1 hy.2.2 with ⟨rfl, rfl⟩
  /-
    case mk.mk.intro.intro
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    x2 y2 : G
    h : UniqueMul A B x2 y2
    hx hy : And (Membership.mem A x2) (And (Membership.mem B y2) (Eq (HMul.hMul x2 …
    ⊢ Eq { fst := x2, snd := y2 } { fst := x2, snd := y2 }
  -/
  rfl
  /-
    🎉 no goals
  -/

-- Porting note: mathport warning: expanding binder collection
--  (ab «expr ∈ » [finset.product/multiset.product/set.prod/list.product](A, B)) -/

@[to_additive]
theorem iff_existsUnique (aA : a0 ∈ A) (bB : b0 ∈ B) :
    UniqueMul A B a0 b0 ↔ ∃! ab, ab ∈ A ×ˢ B ∧ ab.1 * ab.2 = a0 * b0 :=
                                                             /-
                                                               G : Type u_1
                                                               inst✝ : Mul G
                                                               A B : Finset G
                                                               a0 b0 : G
                                                               aA : Membership.mem A a0
                                                               bB : Membership.mem B b0
                                                               x✝ : UniqueMul A B a0 b0
                                                               ⊢ ∀ (y : Prod G G), (fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq ( …
                                                             -/
  ⟨fun _ ↦ ⟨(a0, b0), ⟨Finset.mk_mem_product aA bB, rfl⟩, by simpa⟩,
                                                             /-
                                                               🎉 no goals
                                                             -/
    fun h ↦ h.elim
      (by
        /-
          G : Type u_1
          inst✝ : Mul G
          A B : Finset G
          a0 b0 : G
          aA : Membership.mem A a0
          bB : Membership.mem B b0
          h : ExistsUnique fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq (HMul …
          ⊢ ∀ (x : Prod G G), And (Membership.mem (SProd.sprod A B) x) (Eq (HMul.hMul x. …
        -/
        rintro ⟨x1, x2⟩ _ J x y hx hy l
        /-
          case mk
          G : Type u_1
          inst✝ : Mul G
          A B : Finset G
          a0 b0 : G
          aA : Membership.mem A a0
          bB : Membership.mem B b0
          h : ExistsUnique fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq (HMul …
          x1 x2 : G
          a✝ : And (Membership.mem (SProd.sprod A B) { fst := x1, snd := x2 }) (Eq (HMul …
          J : ∀ (y : Prod G G), And (Membership.mem (SProd.sprod A B) y) (Eq (HMul.hMul  …
          x y : G
          hx : Membership.mem A x
          hy : Membership.mem B y
          l : Eq (HMul.hMul x y) (HMul.hMul a0 b0)
          ⊢ And (Eq x a0) (Eq y b0)
        -/
        rcases Prod.mk.inj_iff.mp (J (a0, b0) ⟨Finset.mk_mem_product aA bB, rfl⟩) with ⟨rfl, rfl⟩
        /-
          case mk.intro
          G : Type u_1
          inst✝ : Mul G
          A B : Finset G
          a0 b0 : G
          aA : Membership.mem A a0
          bB : Membership.mem B b0
          h : ExistsUnique fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq (HMul …
          x y : G
          hx : Membership.mem A x
          hy : Membership.mem B y
          l : Eq (HMul.hMul x y) (HMul.hMul a0 b0)
          a✝ : And (Membership.mem (SProd.sprod A B) { fst := a0, snd := b0 }) (Eq (HMul …
          J : ∀ (y : Prod G G), And (Membership.mem (SProd.sprod A B) y) (Eq (HMul.hMul  …
          ⊢ And (Eq x a0) (Eq y b0)
        -/
        exact Prod.mk.inj_iff.mp (J (x, y) ⟨Finset.mk_mem_product hx hy, l⟩))⟩
        /-
          🎉 no goals
        -/


open Finset in
@[to_additive iff_card_le_one]
theorem iff_card_le_one [DecidableEq G] (ha0 : a0 ∈ A) (hb0 : b0 ∈ B) :
    UniqueMul A B a0 b0 ↔ #{p ∈ A ×ˢ B | p.1 * p.2 = a0 * b0} ≤ 1 := by
  /-
    G : Type u_1
    inst✝¹ : Mul G
    A B : Finset G
    a0 b0 : G
    inst✝ : DecidableEq G
    ha0 : Membership.mem A a0
    hb0 : Membership.mem B b0
    ⊢ Iff (UniqueMul A B a0 b0) (LE.le (Finset.filter (fun p => Eq (HMul.hMul p.1  …
  -/
  simp_rw [card_le_one_iff, mem_filter, mem_product]
  /-
    G : Type u_1
    inst✝¹ : Mul G
    A B : Finset G
    a0 b0 : G
    inst✝ : DecidableEq G
    ha0 : Membership.mem A a0
    hb0 : Membership.mem B b0
    ⊢ Iff (UniqueMul A B a0 b0) (∀ {a b : Prod G G}, And (And (Membership.mem A a. …
  -/
  refine ⟨fun h p1 p2 ⟨⟨ha1, hb1⟩, he1⟩ ⟨⟨ha2, hb2⟩, he2⟩ ↦ ?_, fun h a b ha hb he ↦ ?_⟩
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Mul G
      A B : Finset G
      a0 b0 : G
      inst✝ : DecidableEq G
      ha0 : Membership.mem A a0
      hb0 : Membership.mem B b0
      h : UniqueMul A B a0 b0
      p1 p2 : Prod G G
      x✝¹ : And (And (Membership.mem A p1.1) (Membership.mem B p1.2)) (Eq (HMul.hMul …
      x✝ : And (And (Membership.mem A p2.1) (Membership.mem B p2.2)) (Eq (HMul.hMul  …
      ha1 : Membership.mem A p1.1
      hb1 : Membership.mem B p1.2
      he1 : Eq (HMul.hMul p1.1 p1.2) (HMul.hMul a0 b0)
      ha2 : Membership.mem A p2.1
      hb2 : Membership.mem B p2.2
      he2 : Eq (HMul.hMul p2.1 p2.2) (HMul.hMul a0 b0)
      ⊢ Eq p1 p2
    -/
  · have h1 := h ha1 hb1 he1; have h2 := h ha2 hb2 he2
    /-
      case refine_1
      G : Type u_1
      inst✝¹ : Mul G
      A B : Finset G
      a0 b0 : G
      inst✝ : DecidableEq G
      ha0 : Membership.mem A a0
      hb0 : Membership.mem B b0
      h : UniqueMul A B a0 b0
      p1 p2 : Prod G G
      x✝¹ : And (And (Membership.mem A p1.1) (Membership.mem B p1.2)) (Eq (HMul.hMul …
      x✝ : And (And (Membership.mem A p2.1) (Membership.mem B p2.2)) (Eq (HMul.hMul  …
      ha1 : Membership.mem A p1.1
      hb1 : Membership.mem B p1.2
      he1 : Eq (HMul.hMul p1.1 p1.2) (HMul.hMul a0 b0)
      ha2 : Membership.mem A p2.1
      hb2 : Membership.mem B p2.2
      he2 : Eq (HMul.hMul p2.1 p2.2) (HMul.hMul a0 b0)
      h1 : And (Eq p1.1 a0) (Eq p1.2 b0)
      h2 : And (Eq p2.1 a0) (Eq p2.2 b0)
      ⊢ Eq p1 p2
    -/
    ext
      /-
        case refine_1.fst
        G : Type u_1
        inst✝¹ : Mul G
        A B : Finset G
        a0 b0 : G
        inst✝ : DecidableEq G
        ha0 : Membership.mem A a0
        hb0 : Membership.mem B b0
        h : UniqueMul A B a0 b0
        p1 p2 : Prod G G
        x✝¹ : And (And (Membership.mem A p1.1) (Membership.mem B p1.2)) (Eq (HMul.hMul …
        x✝ : And (And (Membership.mem A p2.1) (Membership.mem B p2.2)) (Eq (HMul.hMul  …
        ha1 : Membership.mem A p1.1
        hb1 : Membership.mem B p1.2
        he1 : Eq (HMul.hMul p1.1 p1.2) (HMul.hMul a0 b0)
        ha2 : Membership.mem A p2.1
        hb2 : Membership.mem B p2.2
        he2 : Eq (HMul.hMul p2.1 p2.2) (HMul.hMul a0 b0)
        h1 : And (Eq p1.1 a0) (Eq p1.2 b0)
        h2 : And (Eq p2.1 a0) (Eq p2.2 b0)
        ⊢ Eq p1.1 p2.1
      -/
    · rw [h1.1, h2.1]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.snd
        G : Type u_1
        inst✝¹ : Mul G
        A B : Finset G
        a0 b0 : G
        inst✝ : DecidableEq G
        ha0 : Membership.mem A a0
        hb0 : Membership.mem B b0
        h : UniqueMul A B a0 b0
        p1 p2 : Prod G G
        x✝¹ : And (And (Membership.mem A p1.1) (Membership.mem B p1.2)) (Eq (HMul.hMul …
        x✝ : And (And (Membership.mem A p2.1) (Membership.mem B p2.2)) (Eq (HMul.hMul  …
        ha1 : Membership.mem A p1.1
        hb1 : Membership.mem B p1.2
        he1 : Eq (HMul.hMul p1.1 p1.2) (HMul.hMul a0 b0)
        ha2 : Membership.mem A p2.1
        hb2 : Membership.mem B p2.2
        he2 : Eq (HMul.hMul p2.1 p2.2) (HMul.hMul a0 b0)
        h1 : And (Eq p1.1 a0) (Eq p1.2 b0)
        h2 : And (Eq p2.1 a0) (Eq p2.2 b0)
        ⊢ Eq p1.2 p2.2
      -/
    · rw [h1.2, h2.2]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      G : Type u_1
      inst✝¹ : Mul G
      A B : Finset G
      a0 b0 : G
      inst✝ : DecidableEq G
      ha0 : Membership.mem A a0
      hb0 : Membership.mem B b0
      h : ∀ {a b : Prod G G}, And (And (Membership.mem A a.1) (Membership.mem B a.2) …
      a b : G
      ha : Membership.mem A a
      hb : Membership.mem B b
      he : Eq (HMul.hMul a b) (HMul.hMul a0 b0)
      ⊢ And (Eq a a0) (Eq b b0)
    -/
  · exact Prod.ext_iff.1 (@h (a, b) (a0, b0) ⟨⟨ha, hb⟩, he⟩ ⟨⟨ha0, hb0⟩, rfl⟩)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-23")]
alias _root_.UniqueAdd.iff_card_nonpos := UniqueAdd.iff_card_le_one

-- Porting note: mathport warning: expanding binder collection
--  (ab «expr ∈ » [finset.product/multiset.product/set.prod/list.product](A, B)) -/

@[to_additive]
theorem exists_iff_exists_existsUnique :
    (∃ a0 b0 : G, a0 ∈ A ∧ b0 ∈ B ∧ UniqueMul A B a0 b0) ↔
      ∃ g : G, ∃! ab, ab ∈ A ×ˢ B ∧ ab.1 * ab.2 = g :=
  ⟨fun ⟨_, _, hA, hB, h⟩ ↦ ⟨_, (iff_existsUnique hA hB).mp h⟩, fun ⟨g, h⟩ ↦ by
    /-
      G : Type u_1
      inst✝ : Mul G
      A B : Finset G
      x✝ : Exists fun g => ExistsUnique fun ab => And (Membership.mem (SProd.sprod A …
      g : G
      h : ExistsUnique fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq (HMul …
      ⊢ Exists fun a0 => Exists fun b0 => And (Membership.mem A a0) (And (Membership …
    -/
    have h' := h
    /-
      G : Type u_1
      inst✝ : Mul G
      A B : Finset G
      x✝ : Exists fun g => ExistsUnique fun ab => And (Membership.mem (SProd.sprod A …
      g : G
      h h' : ExistsUnique fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq (H …
      ⊢ Exists fun a0 => Exists fun b0 => And (Membership.mem A a0) (And (Membership …
    -/
    rcases h' with ⟨⟨a, b⟩, ⟨hab, rfl, -⟩, -⟩
    /-
      case intro.mk.intro.intro.refl
      G : Type u_1
      inst✝ : Mul G
      A B : Finset G
      x✝ : Exists fun g => ExistsUnique fun ab => And (Membership.mem (SProd.sprod A …
      a b : G
      hab : Membership.mem (SProd.sprod A B) { fst := a, snd := b }
      h : ExistsUnique fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq (HMul …
      ⊢ Exists fun a0 => Exists fun b0 => And (Membership.mem A a0) (And (Membership …
    -/
    cases' Finset.mem_product.mp hab with ha hb
    /-
      case intro.mk.intro.intro.refl.intro
      G : Type u_1
      inst✝ : Mul G
      A B : Finset G
      x✝ : Exists fun g => ExistsUnique fun ab => And (Membership.mem (SProd.sprod A …
      a b : G
      hab : Membership.mem (SProd.sprod A B) { fst := a, snd := b }
      h : ExistsUnique fun ab => And (Membership.mem (SProd.sprod A B) ab) (Eq (HMul …
      ha : Membership.mem A { fst := a, snd := b }.1
      hb : Membership.mem B { fst := a, snd := b }.2
      ⊢ Exists fun a0 => Exists fun b0 => And (Membership.mem A a0) (And (Membership …
    -/
    exact ⟨a, b, ha, hb, (iff_existsUnique ha hb).mpr h⟩⟩
    /-
      🎉 no goals
    -/


/-- `UniqueMul` is preserved by inverse images under injective, multiplicative maps. -/
@[to_additive "`UniqueAdd` is preserved by inverse images under injective, additive maps."]
theorem mulHom_preimage (f : G →ₙ* H) (hf : Function.Injective f) (a0 b0 : G) {A B : Finset H}
    (u : UniqueMul A B (f a0) (f b0)) :
    UniqueMul (A.preimage f hf.injOn) (B.preimage f hf.injOn) a0 b0 := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Mul G
    inst✝ : Mul H
    f : MulHom G H
    hf : Function.Injective ⇑f
    a0 b0 : G
    A B : Finset H
    u : UniqueMul A B (f a0) (f b0)
    ⊢ UniqueMul (A.preimage ⇑f ⋯) (B.preimage ⇑f ⋯) a0 b0
  -/
  intro a b ha hb ab
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Mul G
    inst✝ : Mul H
    f : MulHom G H
    hf : Function.Injective ⇑f
    a0 b0 : G
    A B : Finset H
    u : UniqueMul A B (f a0) (f b0)
    a b : G
    ha : Membership.mem (A.preimage ⇑f ⋯) a
    hb : Membership.mem (B.preimage ⇑f ⋯) b
    ab : Eq (HMul.hMul a b) (HMul.hMul a0 b0)
    ⊢ And (Eq a a0) (Eq b b0)
  -/
  simp only [← hf.eq_iff, map_mul] at ab ⊢
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Mul G
    inst✝ : Mul H
    f : MulHom G H
    hf : Function.Injective ⇑f
    a0 b0 : G
    A B : Finset H
    u : UniqueMul A B (f a0) (f b0)
    a b : G
    ha : Membership.mem (A.preimage ⇑f ⋯) a
    hb : Membership.mem (B.preimage ⇑f ⋯) b
    ab : Eq (HMul.hMul (f a) (f b)) (HMul.hMul (f a0) (f b0))
    ⊢ And (Eq (f a) (f a0)) (Eq (f b) (f b0))
  -/
  exact u (Finset.mem_preimage.mp ha) (Finset.mem_preimage.mp hb) ab
  /-
    🎉 no goals
  -/


@[to_additive] theorem of_mulHom_image [DecidableEq H] (f : G →ₙ* H)
    (hf : ∀ ⦃a b c d : G⦄, a * b = c * d → f a = f c ∧ f b = f d → a = c ∧ b = d)
    (h : UniqueMul (A.image f) (B.image f) (f a0) (f b0)) : UniqueMul A B a0 b0 :=
  fun a b ha hb ab ↦ hf ab
                                                                           /-
                                                                             G : Type u_1
                                                                             H : Type u_2
                                                                             inst✝² : Mul G
                                                                             inst✝¹ : Mul H
                                                                             A B : Finset G
                                                                             a0 b0 : G
                                                                             inst✝ : DecidableEq H
                                                                             f : MulHom G H
                                                                             hf : ∀ ⦃a b c d : G⦄, Eq (HMul.hMul a b) (HMul.hMul c d) → And (Eq (f a) (f c) …
                                                                             h : UniqueMul (Finset.image (⇑f) A) (Finset.image (⇑f) B) (f a0) (f b0)
                                                                             a b : G
                                                                             ha : Membership.mem A a
                                                                             hb : Membership.mem B b
                                                                             ab : Eq (HMul.hMul a b) (HMul.hMul a0 b0)
                                                                             ⊢ Eq (HMul.hMul (f a) (f b)) (HMul.hMul (f a0) (f b0))
                                                                           -/
    (h (Finset.mem_image_of_mem f ha) (Finset.mem_image_of_mem f hb) <| by simp_rw [← map_mul, ab])
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


/-- `Unique_Mul` is preserved under multiplicative maps that are injective.

See `UniqueMul.mulHom_map_iff` for a version with swapped bundling. -/
@[to_additive
      "`UniqueAdd` is preserved under additive maps that are injective.

See `UniqueAdd.addHom_map_iff` for a version with swapped bundling."]
theorem mulHom_image_iff [DecidableEq H] (f : G →ₙ* H) (hf : Function.Injective f) :
    UniqueMul (A.image f) (B.image f) (f a0) (f b0) ↔ UniqueMul A B a0 b0 :=
  ⟨of_mulHom_image f fun _ _ _ _ _ ↦ .imp (hf ·) (hf ·), fun h _ _ ↦ by
    /-
      G : Type u_1
      H : Type u_2
      inst✝² : Mul G
      inst✝¹ : Mul H
      A B : Finset G
      a0 b0 : G
      inst✝ : DecidableEq H
      f : MulHom G H
      hf : Function.Injective ⇑f
      h : UniqueMul A B a0 b0
      x✝¹ x✝ : H
      ⊢ Membership.mem (Finset.image (⇑f) A) x✝¹ → Membership.mem (Finset.image (⇑f) …
    -/
    simp_rw [Finset.mem_image]
    /-
      G : Type u_1
      H : Type u_2
      inst✝² : Mul G
      inst✝¹ : Mul H
      A B : Finset G
      a0 b0 : G
      inst✝ : DecidableEq H
      f : MulHom G H
      hf : Function.Injective ⇑f
      h : UniqueMul A B a0 b0
      x✝¹ x✝ : H
      ⊢ (Exists fun a => And (Membership.mem A a) (Eq (f a) x✝¹)) → (Exists fun a => …
    -/
    rintro ⟨a, aA, rfl⟩ ⟨b, bB, rfl⟩ ab
    /-
      case intro.intro.intro.intro
      G : Type u_1
      H : Type u_2
      inst✝² : Mul G
      inst✝¹ : Mul H
      A B : Finset G
      a0 b0 : G
      inst✝ : DecidableEq H
      f : MulHom G H
      hf : Function.Injective ⇑f
      h : UniqueMul A B a0 b0
      a : G
      aA : Membership.mem A a
      b : G
      bB : Membership.mem B b
      ab : Eq (HMul.hMul (f a) (f b)) (HMul.hMul (f a0) (f b0))
      ⊢ And (Eq (f a) (f a0)) (Eq (f b) (f b0))
    -/
    simp_rw [← map_mul, hf.eq_iff] at ab ⊢
    /-
      case intro.intro.intro.intro
      G : Type u_1
      H : Type u_2
      inst✝² : Mul G
      inst✝¹ : Mul H
      A B : Finset G
      a0 b0 : G
      inst✝ : DecidableEq H
      f : MulHom G H
      hf : Function.Injective ⇑f
      h : UniqueMul A B a0 b0
      a : G
      aA : Membership.mem A a
      b : G
      bB : Membership.mem B b
      ab : Eq (HMul.hMul a b) (HMul.hMul a0 b0)
      ⊢ And (Eq a a0) (Eq b b0)
    -/
    exact h aA bB ab⟩
    /-
      🎉 no goals
    -/


/-- `UniqueMul` is preserved under embeddings that are multiplicative.

See `UniqueMul.mulHom_image_iff` for a version with swapped bundling. -/
@[to_additive
      "`UniqueAdd` is preserved under embeddings that are additive.

See `UniqueAdd.addHom_image_iff` for a version with swapped bundling."]
theorem mulHom_map_iff (f : G ↪ H) (mul : ∀ x y, f (x * y) = f x * f y) :
    UniqueMul (A.map f) (B.map f) (f a0) (f b0) ↔ UniqueMul A B a0 b0 := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Mul G
    inst✝ : Mul H
    A B : Finset G
    a0 b0 : G
    f : Function.Embedding G H
    mul : ∀ (x y : G), Eq (f (HMul.hMul x y)) (HMul.hMul (f x) (f y))
    ⊢ Iff (UniqueMul (Finset.map f A) (Finset.map f B) (f a0) (f b0)) (UniqueMul A …
  -/
  classical simp_rw [← mulHom_image_iff ⟨f, mul⟩ f.2, Finset.map_eq_image]; rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem of_mulOpposite
    (h : UniqueMul (B.map ⟨_, op_injective⟩) (A.map ⟨_, op_injective⟩) (op b0) (op a0)) :
    UniqueMul A B a0 b0 := fun a b aA bB ab ↦ by
  /-
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    a0 b0 : G
    h : UniqueMul (Finset.map { toFun := MulOpposite.op, inj' := ⋯ } B) (Finset.ma …
    a b : G
    aA : Membership.mem A a
    bB : Membership.mem B b
    ab : Eq (HMul.hMul a b) (HMul.hMul a0 b0)
    ⊢ And (Eq a a0) (Eq b b0)
  -/
  simpa [and_comm] using h (mem_map_of_mem _ bB) (mem_map_of_mem _ aA) (congr_arg op ab)
  /-
    🎉 no goals
  -/


@[to_additive]
theorem to_mulOpposite (h : UniqueMul A B a0 b0) :
    UniqueMul (B.map ⟨_, op_injective⟩) (A.map ⟨_, op_injective⟩) (op b0) (op a0) :=
                     /-
                       G : Type u_1
                       inst✝ : Mul G
                       A B : Finset G
                       a0 b0 : G
                       h : UniqueMul A B a0 b0
                       ⊢ UniqueMul (Finset.map { toFun := MulOpposite.op, inj' := ⋯ } (Finset.map { t …
                     -/
  of_mulOpposite (by simp_rw [map_map]; exact (mulHom_map_iff _ fun _ _ ↦ by rfl).mpr h)
                                        /-
                                          🎉 no goals
                                        -/


@[to_additive]
theorem iff_mulOpposite :
    UniqueMul (B.map ⟨_, op_injective⟩) (A.map ⟨_, op_injective⟩) (op b0) (op a0) ↔
      UniqueMul A B a0 b0 :=
  ⟨of_mulOpposite, to_mulOpposite⟩


open Finset in
@[to_additive]
theorem of_image_filter [DecidableEq H]
    (f : G →ₙ* H) {A B : Finset G} {aG bG : G} {aH bH : H} (hae : f aG = aH) (hbe : f bG = bH)
    (huH : UniqueMul (A.image f) (B.image f) aH bH)
    (huG : UniqueMul {a ∈ A | f a = aH} {b ∈ B | f b = bH} aG bG) :
    UniqueMul A B aG bG := fun a b ha hb he ↦ by
  /-
    G : Type u_1
    H : Type u_2
    inst✝² : Mul G
    inst✝¹ : Mul H
    inst✝ : DecidableEq H
    f : MulHom G H
    A B : Finset G
    aG bG : G
    aH bH : H
    hae : Eq (f aG) aH
    hbe : Eq (f bG) bH
    huH : UniqueMul (Finset.image (⇑f) A) (Finset.image (⇑f) B) aH bH
    huG : UniqueMul (Finset.filter (fun a => Eq (f a) aH) A) (Finset.filter (fun b …
    a b : G
    ha : Membership.mem A a
    hb : Membership.mem B b
    he : Eq (HMul.hMul a b) (HMul.hMul aG bG)
    ⊢ And (Eq a aG) (Eq b bG)
  -/
  specialize huH (mem_image_of_mem _ ha) (mem_image_of_mem _ hb)
  /-
    G : Type u_1
    H : Type u_2
    inst✝² : Mul G
    inst✝¹ : Mul H
    inst✝ : DecidableEq H
    f : MulHom G H
    A B : Finset G
    aG bG : G
    aH bH : H
    hae : Eq (f aG) aH
    hbe : Eq (f bG) bH
    huG : UniqueMul (Finset.filter (fun a => Eq (f a) aH) A) (Finset.filter (fun b …
    a b : G
    ha : Membership.mem A a
    hb : Membership.mem B b
    he : Eq (HMul.hMul a b) (HMul.hMul aG bG)
    huH : Eq (HMul.hMul (f a) (f b)) (HMul.hMul aH bH) → And (Eq (f a) aH) (Eq (f  …
    ⊢ And (Eq a aG) (Eq b bG)
  -/
  rw [← map_mul, he, map_mul, hae, hbe] at huH
  /-
    G : Type u_1
    H : Type u_2
    inst✝² : Mul G
    inst✝¹ : Mul H
    inst✝ : DecidableEq H
    f : MulHom G H
    A B : Finset G
    aG bG : G
    aH bH : H
    hae : Eq (f aG) aH
    hbe : Eq (f bG) bH
    huG : UniqueMul (Finset.filter (fun a => Eq (f a) aH) A) (Finset.filter (fun b …
    a b : G
    ha : Membership.mem A a
    hb : Membership.mem B b
    he : Eq (HMul.hMul a b) (HMul.hMul aG bG)
    huH : Eq (HMul.hMul aH bH) (HMul.hMul aH bH) → And (Eq (f a) aH) (Eq (f b) bH)
    ⊢ And (Eq a aG) (Eq b bG)
  -/
  refine huG ?_ ?_ he <;> rw [mem_filter]
  /-
    case refine_1
    G : Type u_1
    H : Type u_2
    inst✝² : Mul G
    inst✝¹ : Mul H
    inst✝ : DecidableEq H
    f : MulHom G H
    A B : Finset G
    aG bG : G
    aH bH : H
    hae : Eq (f aG) aH
    hbe : Eq (f bG) bH
    huG : UniqueMul (Finset.filter (fun a => Eq (f a) aH) A) (Finset.filter (fun b …
    a b : G
    ha : Membership.mem A a
    hb : Membership.mem B b
    he : Eq (HMul.hMul a b) (HMul.hMul aG bG)
    huH : Eq (HMul.hMul aH bH) (HMul.hMul aH bH) → And (Eq (f a) aH) (Eq (f b) bH)
    ⊢ And (Membership.mem A a) (Eq (f a) aH)
  -/
  exacts [⟨ha, (huH rfl).1⟩, ⟨hb, (huH rfl).2⟩]
  /-
    🎉 no goals
  -/


/-- Let `G` be a Type with addition.  `UniqueSums G` asserts that any two non-empty
finite subsets of `G` have the `UniqueAdd` property, with respect to some element of their
sum `A + B`. -/
class UniqueSums (G) [Add G] : Prop where
/-- For `A B` two nonempty finite sets, there always exist `a0 ∈ A, b0 ∈ B` such that
`UniqueAdd A B a0 b0` -/
  uniqueAdd_of_nonempty :
    ∀ {A B : Finset G}, A.Nonempty → B.Nonempty → ∃ a0 ∈ A, ∃ b0 ∈ B, UniqueAdd A B a0 b0


/-- Let `G` be a Type with multiplication.  `UniqueProds G` asserts that any two non-empty
finite subsets of `G` have the `UniqueMul` property, with respect to some element of their
product `A * B`. -/
class UniqueProds (G) [Mul G] : Prop where
/-- For `A B` two nonempty finite sets, there always exist `a0 ∈ A, b0 ∈ B` such that
`UniqueMul A B a0 b0` -/
  uniqueMul_of_nonempty :
    ∀ {A B : Finset G}, A.Nonempty → B.Nonempty → ∃ a0 ∈ A, ∃ b0 ∈ B, UniqueMul A B a0 b0


/-- Let `G` be a Type with addition. `TwoUniqueSums G` asserts that any two non-empty
finite subsets of `G`, at least one of which is not a singleton, possesses at least two pairs
of elements satisfying the `UniqueAdd` property. -/
class TwoUniqueSums (G) [Add G] : Prop where
/-- For `A B` two finite sets whose product has cardinality at least 2,
  we can find at least two unique pairs. -/
  uniqueAdd_of_one_lt_card : ∀ {A B : Finset G}, 1 < #A * #B →
    ∃ p1 ∈ A ×ˢ B, ∃ p2 ∈ A ×ˢ B, p1 ≠ p2 ∧ UniqueAdd A B p1.1 p1.2 ∧ UniqueAdd A B p2.1 p2.2


/-- Let `G` be a Type with multiplication. `TwoUniqueProds G` asserts that any two non-empty
finite subsets of `G`, at least one of which is not a singleton, possesses at least two pairs
of elements satisfying the `UniqueMul` property. -/
class TwoUniqueProds (G) [Mul G] : Prop where
/-- For `A B` two finite sets whose product has cardinality at least 2,
  we can find at least two unique pairs. -/
  uniqueMul_of_one_lt_card : ∀ {A B : Finset G}, 1 < #A * #B →
    ∃ p1 ∈ A ×ˢ B, ∃ p2 ∈ A ×ˢ B, p1 ≠ p2 ∧ UniqueMul A B p1.1 p1.2 ∧ UniqueMul A B p2.1 p2.2


@[to_additive]
lemma uniqueMul_of_twoUniqueMul {G} [Mul G] {A B : Finset G} (h : 1 < #A * #B →
    ∃ p1 ∈ A ×ˢ B, ∃ p2 ∈ A ×ˢ B, p1 ≠ p2 ∧ UniqueMul A B p1.1 p1.2 ∧ UniqueMul A B p2.1 p2.2)
    (hA : A.Nonempty) (hB : B.Nonempty) : ∃ a ∈ A, ∃ b ∈ B, UniqueMul A B a b := by
  /-
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    h : LT.lt 1 (HMul.hMul A.card B.card) → Exists fun p1 => And (Membership.mem ( …
    hA : A.Nonempty
    hB : B.Nonempty
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  by_cases hc : #A ≤ 1 ∧ #B ≤ 1
    /-
      case pos
      G : Type u_1
      inst✝ : Mul G
      A B : Finset G
      h : LT.lt 1 (HMul.hMul A.card B.card) → Exists fun p1 => And (Membership.mem ( …
      hA : A.Nonempty
      hB : B.Nonempty
      hc : And (LE.le A.card 1) (LE.le B.card 1)
      ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
    -/
  · exact UniqueMul.of_card_le_one hA hB hc.1 hc.2
    /-
      🎉 no goals
    -/
  /-
    case neg
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    h : LT.lt 1 (HMul.hMul A.card B.card) → Exists fun p1 => And (Membership.mem ( …
    hA : A.Nonempty
    hB : B.Nonempty
    hc : Not (And (LE.le A.card 1) (LE.le B.card 1))
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  simp_rw [not_and_or, not_le] at hc
  /-
    case neg
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    h : LT.lt 1 (HMul.hMul A.card B.card) → Exists fun p1 => And (Membership.mem ( …
    hA : A.Nonempty
    hB : B.Nonempty
    hc : Or (LT.lt 1 A.card) (LT.lt 1 B.card)
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  rw [← Finset.card_pos] at hA hB
  /-
    case neg
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    h : LT.lt 1 (HMul.hMul A.card B.card) → Exists fun p1 => And (Membership.mem ( …
    hA : LT.lt 0 A.card
    hB : LT.lt 0 B.card
    hc : Or (LT.lt 1 A.card) (LT.lt 1 B.card)
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  obtain ⟨p, hp, _, _, _, hu, _⟩ := h (Nat.one_lt_mul_iff.mpr ⟨hA, hB, hc⟩)
  /-
    case neg.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    h : LT.lt 1 (HMul.hMul A.card B.card) → Exists fun p1 => And (Membership.mem ( …
    hA : LT.lt 0 A.card
    hB : LT.lt 0 B.card
    hc : Or (LT.lt 1 A.card) (LT.lt 1 B.card)
    p : Prod G G
    hp : Membership.mem (SProd.sprod A B) p
    w✝ : Prod G G
    left✝¹ : Membership.mem (SProd.sprod A B) w✝
    left✝ : Ne p w✝
    hu : UniqueMul A B p.1 p.2
    right✝ : UniqueMul A B w✝.1 w✝.2
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  rw [Finset.mem_product] at hp
  /-
    case neg.intro.intro.intro.intro.intro.intro
    G : Type u_1
    inst✝ : Mul G
    A B : Finset G
    h : LT.lt 1 (HMul.hMul A.card B.card) → Exists fun p1 => And (Membership.mem ( …
    hA : LT.lt 0 A.card
    hB : LT.lt 0 B.card
    hc : Or (LT.lt 1 A.card) (LT.lt 1 B.card)
    p : Prod G G
    hp : And (Membership.mem A p.1) (Membership.mem B p.2)
    w✝ : Prod G G
    left✝¹ : Membership.mem (SProd.sprod A B) w✝
    left✝ : Ne p w✝
    hu : UniqueMul A B p.1 p.2
    right✝ : UniqueMul A B w✝.1 w✝.2
    ⊢ Exists fun a => And (Membership.mem A a) (Exists fun b => And (Membership.me …
  -/
  exact ⟨p.1, hp.1, p.2, hp.2, hu⟩
  /-
    🎉 no goals
  -/


@[to_additive] instance TwoUniqueProds.toUniqueProds (G) [Mul G] [TwoUniqueProds G] :
    UniqueProds G where
  uniqueMul_of_nonempty := uniqueMul_of_twoUniqueMul uniqueMul_of_one_lt_card


instance {M} [Add M] [UniqueSums M] : UniqueProds (Multiplicative M) where
  uniqueMul_of_nonempty := UniqueSums.uniqueAdd_of_nonempty (G := M)


instance {M} [Add M] [TwoUniqueSums M] : TwoUniqueProds (Multiplicative M) where
  uniqueMul_of_one_lt_card := TwoUniqueSums.uniqueAdd_of_one_lt_card (G := M)


instance {M} [Mul M] [UniqueProds M] : UniqueSums (Additive M) where
  uniqueAdd_of_nonempty := UniqueProds.uniqueMul_of_nonempty (G := M)


instance {M} [Mul M] [TwoUniqueProds M] : TwoUniqueSums (Additive M) where
  uniqueAdd_of_one_lt_card := TwoUniqueProds.uniqueMul_of_one_lt_card (G := M)


private abbrev I : Bool → Type max u v := Bool.rec (ULift.{v} G) (ULift.{u} H)

@[to_additive] private instance : ∀ b, Mul (I G H b) := Bool.rec ULift.mul ULift.mul

@[to_additive] private def Prod.upMulHom : G × H →ₙ* ∀ b, I G H b :=
                                              /-
                                                G : Type u
                                                H : Type v
                                                inst✝¹ : Mul G
                                                inst✝ : Mul H
                                                x y : Prod G H
                                                ⊢ Eq ((fun x t => Bool.rec { down := x.1 } { down := x.2 } t) (HMul.hMul x y)) …
                                              -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  ⟨fun x ↦ Bool.rec ⟨x.1⟩ ⟨x.2⟩, fun x y ↦ by ext (_|_) <;> rfl⟩
                                                            /-
                                                              🎉 no goals
                                                            -/

@[to_additive] private def downMulHom : ULift G →ₙ* G := ⟨ULift.down, fun _ _ ↦ rfl⟩


@[to_additive] theorem of_mulHom (f : H →ₙ* G)
    (hf : ∀ ⦃a b c d : H⦄, a * b = c * d → f a = f c ∧ f b = f d → a = c ∧ b = d)
    [UniqueProds G] : UniqueProds H where
  uniqueMul_of_nonempty {A B} A0 B0 := by
    classical
    obtain ⟨a0, ha0, b0, hb0, h⟩ := uniqueMul_of_nonempty (A0.image f) (B0.image f)
    obtain ⟨a', ha', rfl⟩ := mem_image.mp ha0
    obtain ⟨b', hb', rfl⟩ := mem_image.mp hb0
    exact ⟨a', ha', b', hb', UniqueMul.of_mulHom_image f hf h⟩


@[to_additive]
theorem of_injective_mulHom (f : H →ₙ* G) (hf : Function.Injective f) (_ : UniqueProds G) :
    UniqueProds H := of_mulHom f (fun _ _ _ _ _ ↦ .imp (hf ·) (hf ·))


/-- `UniqueProd` is preserved under multiplicative equivalences. -/
@[to_additive "`UniqueSums` is preserved under additive equivalences."]
theorem _root_.MulEquiv.uniqueProds_iff (f : G ≃* H) : UniqueProds G ↔ UniqueProds H :=
  ⟨of_injective_mulHom f.symm f.symm.injective, of_injective_mulHom f f.injective⟩


open Finset MulOpposite in
@[to_additive]
theorem of_mulOpposite (h : UniqueProds Gᵐᵒᵖ) : UniqueProds G where
  uniqueMul_of_nonempty hA hB :=
    let f : G ↪ Gᵐᵒᵖ := ⟨op, op_injective⟩
    let ⟨y, yB, x, xA, hxy⟩ := h.uniqueMul_of_nonempty (hB.map (f := f)) (hA.map (f := f))
    ⟨unop x, (mem_map' _).mp xA, unop y, (mem_map' _).mp yB, hxy.of_mulOpposite⟩


@[to_additive] instance [h : UniqueProds G] : UniqueProds Gᵐᵒᵖ :=
  of_mulOpposite <| (MulEquiv.opOp G).uniqueProds_iff.mp h


@[to_additive] private theorem toIsLeftCancelMul [UniqueProds G] : IsLeftCancelMul G where
  mul_left_cancel a b1 b2 he := by
    classical
    have := mem_insert_self b1 {b2}
    obtain ⟨a, ha, b, hb, hu⟩ := uniqueMul_of_nonempty ⟨a, mem_singleton_self a⟩ ⟨b1, this⟩
    cases mem_singleton.mp ha
    simp_rw [mem_insert, mem_singleton] at hb
    obtain rfl | rfl := hb
    · exact (hu ha (mem_insert_of_mem <| mem_singleton_self b2) he.symm).2.symm
    · exact (hu ha this he).2


open MulOpposite in
@[to_additive] theorem toIsCancelMul [UniqueProds G] : IsCancelMul G where
  mul_left_cancel := toIsLeftCancelMul.mul_left_cancel
  mul_right_cancel _ _ _ h :=
    op_injective <| toIsLeftCancelMul.mul_left_cancel _ _ _ <| unop_injective h


/-- `UniqueProds G` says that for any two nonempty `Finset`s `A` and `B` in `G`, `A × B`
  contains a unique pair with the `UniqueMul` property. Strojnowski showed that if `G` is
  a group, then we only need to check this when `A = B`.
  Here we generalize the result to cancellative semigroups.
  Non-cancellative counterexample: the AddMonoid {0,1} with 1+1=1. -/
@[to_additive] theorem of_same {G} [Semigroup G] [IsCancelMul G]
    (h : ∀ {A : Finset G}, A.Nonempty → ∃ a1 ∈ A, ∃ a2 ∈ A, UniqueMul A A a1 a2) :
    UniqueProds G where
  uniqueMul_of_nonempty {A B} hA hB := by
    classical
    obtain ⟨g1, h1, g2, h2, hu⟩ := h (hB.mul hA)
    obtain ⟨b1, hb1, a1, ha1, rfl⟩ := mem_mul.mp h1
    obtain ⟨b2, hb2, a2, ha2, rfl⟩ := mem_mul.mp h2
    refine ⟨a1, ha1, b2, hb2, fun a b ha hb he => ?_⟩
    specialize hu (mul_mem_mul hb1 ha) (mul_mem_mul hb ha2) _
    · rw [mul_assoc b1, ← mul_assoc a, he, mul_assoc a1, ← mul_assoc b1]
    exact ⟨mul_left_cancel hu.1, mul_right_cancel hu.2⟩


/-- If a group has `UniqueProds`, then it actually has `TwoUniqueProds`.
  For an example of a semigroup `G` embeddable into a group that has `UniqueProds`
  but not `TwoUniqueProds`, see Example 10.13 in
  [J. Okniński, *Semigroup Algebras*][Okninski1991]. -/
@[to_additive] theorem toTwoUniqueProds_of_group {G}
    [Group G] [UniqueProds G] : TwoUniqueProds G where
  uniqueMul_of_one_lt_card {A B} hc := by
    /-
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : UniqueProds G
      A B : Finset G
      hc : LT.lt 1 (HMul.hMul A.card B.card)
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    simp_rw [Nat.one_lt_mul_iff, card_pos] at hc
    /-
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : UniqueProds G
      A B : Finset G
      hc : And A.Nonempty (And B.Nonempty (Or (LT.lt 1 A.card) (LT.lt 1 B.card)))
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    obtain ⟨a, ha, b, hb, hu⟩ := uniqueMul_of_nonempty hc.1 hc.2.1
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : UniqueProds G
      A B : Finset G
      hc : And A.Nonempty (And B.Nonempty (Or (LT.lt 1 A.card) (LT.lt 1 B.card)))
      a : G
      ha : Membership.mem A a
      b : G
      hb : Membership.mem B b
      hu : UniqueMul A B a b
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    let C := A.map ⟨_, mul_right_injective a⁻¹⟩ -- C = a⁻¹A
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : UniqueProds G
      A B : Finset G
      hc : And A.Nonempty (And B.Nonempty (Or (LT.lt 1 A.card) (LT.lt 1 B.card)))
      a : G
      ha : Membership.mem A a
      b : G
      hb : Membership.mem B b
      hu : UniqueMul A B a b
      C : Finset G := Finset.map { toFun := fun x => HMul.hMul (Inv.inv a) x, inj' : …
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    let D := B.map ⟨_, mul_left_injective b⁻¹⟩  -- D = Bb⁻¹
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : UniqueProds G
      A B : Finset G
      hc : And A.Nonempty (And B.Nonempty (Or (LT.lt 1 A.card) (LT.lt 1 B.card)))
      a : G
      ha : Membership.mem A a
      b : G
      hb : Membership.mem B b
      hu : UniqueMul A B a b
      C : Finset G := Finset.map { toFun := fun x => HMul.hMul (Inv.inv a) x, inj' : …
      D : Finset G := Finset.map { toFun := fun x => HMul.hMul x (Inv.inv b), inj' : …
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    have hcard : 1 < #C ∨ 1 < #D := by simp_rw [C, D, card_map]; exact hc.2.2
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : UniqueProds G
      A B : Finset G
      hc : And A.Nonempty (And B.Nonempty (Or (LT.lt 1 A.card) (LT.lt 1 B.card)))
      a : G
      ha : Membership.mem A a
      b : G
      hb : Membership.mem B b
      hu : UniqueMul A B a b
      C : Finset G := Finset.map { toFun := fun x => HMul.hMul (Inv.inv a) x, inj' : …
      D : Finset G := Finset.map { toFun := fun x => HMul.hMul x (Inv.inv b), inj' : …
      hcard : Or (LT.lt 1 C.card) (LT.lt 1 D.card)
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    have hC : 1 ∈ C := mem_map.mpr ⟨a, ha, inv_mul_cancel a⟩
    /-
      case intro.intro.intro.intro
      G : Type u_1
      inst✝¹ : Group G
      inst✝ : UniqueProds G
      A B : Finset G
      hc : And A.Nonempty (And B.Nonempty (Or (LT.lt 1 A.card) (LT.lt 1 B.card)))
      a : G
      ha : Membership.mem A a
      b : G
      hb : Membership.mem B b
      hu : UniqueMul A B a b
      C : Finset G := Finset.map { toFun := fun x => HMul.hMul (Inv.inv a) x, inj' : …
      D : Finset G := Finset.map { toFun := fun x => HMul.hMul x (Inv.inv b), inj' : …
      hcard : Or (LT.lt 1 C.card) (LT.lt 1 D.card)
      hC : Membership.mem C 1
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    have hD : 1 ∈ D := mem_map.mpr ⟨b, hb, mul_inv_cancel b⟩
    suffices ∃ c ∈ C, ∃ d ∈ D, (c ≠ 1 ∨ d ≠ 1) ∧ UniqueMul C D c d by
      simp_rw [mem_product]
      obtain ⟨c, hc, d, hd, hne, hu'⟩ := this
      obtain ⟨a0, ha0, rfl⟩ := mem_map.mp hc
      obtain ⟨b0, hb0, rfl⟩ := mem_map.mp hd
      refine ⟨(_, _), ⟨ha0, hb0⟩, (a, b), ⟨ha, hb⟩, ?_, fun a' b' ha' hb' he => ?_, hu⟩
      · simp_rw [Function.Embedding.coeFn_mk, Ne, inv_mul_eq_one, mul_inv_eq_one] at hne
        rwa [Ne, Prod.mk.inj_iff, not_and_or, eq_comm]
      specialize hu' (mem_map_of_mem _ ha') (mem_map_of_mem _ hb')
      simp_rw [Function.Embedding.coeFn_mk, mul_left_cancel_iff, mul_right_cancel_iff] at hu'
      rw [mul_assoc, ← mul_assoc a', he, mul_assoc, mul_assoc] at hu'
      exact hu' rfl
    classical
    let _ := Finset.mul (α := G)              -- E = D⁻¹C, F = DC⁻¹
    have := uniqueMul_of_nonempty (A := D.image (·⁻¹) * C) (B := D * C.image (·⁻¹)) ?_ ?_
    · obtain ⟨e, he, f, hf, hu⟩ := this
      clear_value C D
      simp only [UniqueMul, mem_mul, mem_image] at he hf hu
      obtain ⟨_, ⟨d1, hd1, rfl⟩, c1, hc1, rfl⟩ := he
      obtain ⟨d2, hd2, _, ⟨c2, hc2, rfl⟩, rfl⟩ := hf
      by_cases h12 : c1 ≠ 1 ∨ d2 ≠ 1
      · refine ⟨c1, hc1, d2, hd2, h12, fun c3 d3 hc3 hd3 he => ?_⟩
        specialize hu ⟨_, ⟨_, hd1, rfl⟩, _, hc3, rfl⟩ ⟨_, hd3, _, ⟨_, hc2, rfl⟩, rfl⟩
        rw [mul_left_cancel_iff, mul_right_cancel_iff,
            mul_assoc, ← mul_assoc c3, he, mul_assoc, mul_assoc] at hu; exact hu rfl
      push_neg at h12; obtain ⟨rfl, rfl⟩ := h12
      by_cases h21 : c2 ≠ 1 ∨ d1 ≠ 1
      · refine ⟨c2, hc2, d1, hd1, h21, fun c4 d4 hc4 hd4 he => ?_⟩
        specialize hu ⟨_, ⟨_, hd4, rfl⟩, _, hC, rfl⟩ ⟨_, hD, _, ⟨_, hc4, rfl⟩, rfl⟩
        simpa only [mul_one, one_mul, ← mul_inv_rev, he, true_imp_iff, inv_inj, and_comm] using hu
      push_neg at h21; obtain ⟨rfl, rfl⟩ := h21
      rcases hcard with hC | hD
      · obtain ⟨c, hc, hc1⟩ := exists_ne_of_one_lt_card hC 1
        refine (hc1 ?_).elim
        simpa using hu ⟨_, ⟨_, hD, rfl⟩, _, hc, rfl⟩ ⟨_, hD, _, ⟨_, hc, rfl⟩, rfl⟩
      · obtain ⟨d, hd, hd1⟩ := exists_ne_of_one_lt_card hD 1
        refine (hd1 ?_).elim
        simpa using hu ⟨_, ⟨_, hd, rfl⟩, _, hC, rfl⟩ ⟨_, hd, _, ⟨_, hC, rfl⟩, rfl⟩
    all_goals apply_rules [Nonempty.mul, Nonempty.image, Finset.Nonempty.map, hc.1, hc.2.1]


open UniqueMul in
@[to_additive] instance instForall {ι} (G : ι → Type*) [∀ i, Mul (G i)] [∀ i, UniqueProds (G i)] :
    UniqueProds (∀ i, G i) where
  uniqueMul_of_nonempty {A} := by
    classical
    let _ := isWellFounded_ssubset (α := ∀ i, G i) -- why need this?
    apply IsWellFounded.induction (· ⊂ ·) A; intro A ihA B hA
    apply IsWellFounded.induction (· ⊂ ·) B; intro B ihB hB
    by_cases hc : #A ≤ 1 ∧ #B ≤ 1
    · exact of_card_le_one hA hB hc.1 hc.2
    simp_rw [not_and_or, not_le] at hc
    obtain ⟨i, hc⟩ := exists_or.mpr (hc.imp exists_of_one_lt_card_pi exists_of_one_lt_card_pi)
    obtain ⟨ai, hA, bi, hB, hi⟩ := uniqueMul_of_nonempty (hA.image (· i)) (hB.image (· i))
    rw [mem_image, ← filter_nonempty_iff] at hA hB
    let A' := {a ∈ A | a i = ai}; let B' := {b ∈ B | b i = bi}
    obtain ⟨a0, ha0, b0, hb0, hu⟩ : ∃ a0 ∈ A', ∃ b0 ∈ B', UniqueMul A' B' a0 b0 := by
      rcases hc with hc | hc; · exact ihA A' (hc.2 ai) hA hB
      by_cases hA' : A' = A
      · rw [hA']
        exact ihB B' (hc.2 bi) hB
      · exact ihA A' ((A.filter_subset _).ssubset_of_ne hA') hA hB
    rw [mem_filter] at ha0 hb0
    exact ⟨a0, ha0.1, b0, hb0.1, of_image_filter (Pi.evalMulHom G i) ha0.2 hb0.2 hi hu⟩


open ULift in
@[to_additive] instance [UniqueProds G] [UniqueProds H] : UniqueProds (G × H) := by
  /-
    G : Type u
    H : Type v
    inst✝³ : Mul G
    inst✝² : Mul H
    inst✝¹ : UniqueProds G
    inst✝ : UniqueProds H
    ⊢ UniqueProds (Prod G H)
  -/
  have : ∀ b, UniqueProds (I G H b) := Bool.rec ?_ ?_
    /-
      case refine_3
      G : Type u
      H : Type v
      inst✝³ : Mul G
      inst✝² : Mul H
      inst✝¹ : UniqueProds G
      inst✝ : UniqueProds H
      ⊢ UniqueProds (I G H Bool.true)
    -/
  · exact of_injective_mulHom (downMulHom H) down_injective ‹_›
    /-
      🎉 no goals
    -/
  · refine of_injective_mulHom (Prod.upMulHom G H) (fun x y he => Prod.ext ?_ ?_)
                                            /-
                                              case refine_1.refine_1
                                              G : Type u
                                              H : Type v
                                              inst✝³ : Mul G
                                              inst✝² : Mul H
                                              inst✝¹ : UniqueProds G
                                              inst✝ : UniqueProds H
                                              this : ∀ (b : Bool), UniqueProds (I G H b)
                                              x y : Prod G H
                                              he : Eq ((Prod.upMulHom G H) x) ((Prod.upMulHom G H) y)
                                              ⊢ Eq x.1 y.1
                                            -/
      (UniqueProds.instForall <| I G H) <;> apply up_injective
    /-
      case refine_1.refine_1.a
      G : Type u
      H : Type v
      inst✝³ : Mul G
      inst✝² : Mul H
      inst✝¹ : UniqueProds G
      inst✝ : UniqueProds H
      this : ∀ (b : Bool), UniqueProds (I G H b)
      x y : Prod G H
      he : Eq ((Prod.upMulHom G H) x) ((Prod.upMulHom G H) y)
      ⊢ Eq { down := x.1 } { down := y.1 }
    -/
    exacts [congr_fun he false, congr_fun he true]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u
      H : Type v
      inst✝³ : Mul G
      inst✝² : Mul H
      inst✝¹ : UniqueProds G
      inst✝ : UniqueProds H
      ⊢ UniqueProds (I G H Bool.false)
    -/
  · exact of_injective_mulHom (downMulHom G) down_injective ‹_›
    /-
      🎉 no goals
    -/


instance {ι} (G : ι → Type*) [∀ i, AddZeroClass (G i)] [∀ i, UniqueSums (G i)] :
    UniqueSums (Π₀ i, G i) :=
  UniqueSums.of_injective_addHom
    DFinsupp.coeFnAddMonoidHom.toAddHom DFunLike.coe_injective inferInstance


instance {ι G} [AddZeroClass G] [UniqueSums G] : UniqueSums (ι →₀ G) :=
  UniqueSums.of_injective_addHom
    Finsupp.coeFnAddHom.toAddHom DFunLike.coe_injective inferInstance


@[to_additive] theorem of_mulHom (f : H →ₙ* G)
    (hf : ∀ ⦃a b c d : H⦄, a * b = c * d → f a = f c ∧ f b = f d → a = c ∧ b = d)
    [TwoUniqueProds G] : TwoUniqueProds H where
  uniqueMul_of_one_lt_card {A B} hc := by
    classical
    obtain hc' | hc' := lt_or_le 1 (#(A.image f) * #(B.image f))
    · obtain ⟨⟨a1, b1⟩, h1, ⟨a2, b2⟩, h2, hne, hu1, hu2⟩ := uniqueMul_of_one_lt_card hc'
      simp_rw [mem_product, mem_image] at h1 h2 ⊢
      obtain ⟨⟨a1, ha1, rfl⟩, b1, hb1, rfl⟩ := h1
      obtain ⟨⟨a2, ha2, rfl⟩, b2, hb2, rfl⟩ := h2
      exact ⟨(a1, b1), ⟨ha1, hb1⟩, (a2, b2), ⟨ha2, hb2⟩, mt (congr_arg (Prod.map f f)) hne,
        UniqueMul.of_mulHom_image f hf hu1, UniqueMul.of_mulHom_image f hf hu2⟩
    rw [← card_product] at hc hc'
    obtain ⟨p1, h1, p2, h2, hne⟩ := one_lt_card_iff_nontrivial.mp hc
    refine ⟨p1, h1, p2, h2, hne, ?_⟩
    cases mem_product.mp h1; cases mem_product.mp h2
    constructor <;> refine UniqueMul.of_mulHom_image f hf
      ((UniqueMul.iff_card_le_one ?_ ?_).mpr <| (card_filter_le _ _).trans hc') <;>
    apply mem_image_of_mem <;> assumption


@[to_additive]
theorem of_injective_mulHom (f : H →ₙ* G) (hf : Function.Injective f)
    (_ : TwoUniqueProds G) : TwoUniqueProds H :=
  of_mulHom f (fun _ _ _ _ _ ↦ .imp (hf ·) (hf ·))


/-- `TwoUniqueProd` is preserved under multiplicative equivalences. -/
@[to_additive "`TwoUniqueSums` is preserved under additive equivalences."]
theorem _root_.MulEquiv.twoUniqueProds_iff (f : G ≃* H) : TwoUniqueProds G ↔ TwoUniqueProds H :=
  ⟨of_injective_mulHom f.symm f.symm.injective, of_injective_mulHom f f.injective⟩


@[to_additive]
instance instForall {ι} (G : ι → Type*) [∀ i, Mul (G i)] [∀ i, TwoUniqueProds (G i)] :
    TwoUniqueProds (∀ i, G i) where
  uniqueMul_of_one_lt_card {A} := by
    classical
    let _ := isWellFounded_ssubset (α := ∀ i, G i) -- why need this?
    apply IsWellFounded.induction (· ⊂ ·) A; intro A ihA B
    apply IsWellFounded.induction (· ⊂ ·) B; intro B ihB hc
    obtain ⟨hA, hB, hc⟩ := Nat.one_lt_mul_iff.mp hc
    rw [card_pos] at hA hB
    obtain ⟨i, hc⟩ := exists_or.mpr (hc.imp exists_of_one_lt_card_pi exists_of_one_lt_card_pi)
    obtain ⟨p1, h1, p2, h2, hne, hi1, hi2⟩ := uniqueMul_of_one_lt_card (Nat.one_lt_mul_iff.mpr
      ⟨card_pos.2 (hA.image _), card_pos.2 (hB.image _), hc.imp And.left And.left⟩)
    simp_rw [mem_product, mem_image, ← filter_nonempty_iff] at h1 h2
    replace h1 := uniqueMul_of_twoUniqueMul ?_ h1.1 h1.2
    on_goal 1 => replace h2 := uniqueMul_of_twoUniqueMul ?_ h2.1 h2.2

    · obtain ⟨a1, ha1, b1, hb1, hu1⟩ := h1
      obtain ⟨a2, ha2, b2, hb2, hu2⟩ := h2
      rw [mem_filter] at ha1 hb1 ha2 hb2
      simp_rw [mem_product]
      refine ⟨(a1, b1), ⟨ha1.1, hb1.1⟩, (a2, b2), ⟨ha2.1, hb2.1⟩, ?_,
        UniqueMul.of_image_filter (Pi.evalMulHom G i) ha1.2 hb1.2 hi1 hu1,
        UniqueMul.of_image_filter (Pi.evalMulHom G i) ha2.2 hb2.2 hi2 hu2⟩
      contrapose! hne; rw [Prod.mk.inj_iff] at hne ⊢
      rw [← ha1.2, ← hb1.2, ← ha2.2, ← hb2.2, hne.1, hne.2]; exact ⟨rfl, rfl⟩
    all_goals rcases hc with hc | hc; · exact ihA _ (hc.2 _)
    · by_cases hA : {a ∈ A | a i = p2.1} = A
      · rw [hA]
        exact ihB _ (hc.2 _)
      · exact ihA _ ((A.filter_subset _).ssubset_of_ne hA)
    · by_cases hA : {a ∈ A | a i = p1.1} = A
      · rw [hA]
        exact ihB _ (hc.2 _)
      · exact ihA _ ((A.filter_subset _).ssubset_of_ne hA)


open ULift in
@[to_additive] instance [TwoUniqueProds G] [TwoUniqueProds H] : TwoUniqueProds (G × H) := by
  /-
    G : Type u
    H : Type v
    inst✝³ : Mul G
    inst✝² : Mul H
    inst✝¹ : TwoUniqueProds G
    inst✝ : TwoUniqueProds H
    ⊢ TwoUniqueProds (Prod G H)
  -/
  have : ∀ b, TwoUniqueProds (I G H b) := Bool.rec ?_ ?_
    /-
      case refine_3
      G : Type u
      H : Type v
      inst✝³ : Mul G
      inst✝² : Mul H
      inst✝¹ : TwoUniqueProds G
      inst✝ : TwoUniqueProds H
      ⊢ TwoUniqueProds (I G H Bool.true)
    -/
  · exact of_injective_mulHom (downMulHom H) down_injective ‹_›
    /-
      🎉 no goals
    -/
  · refine of_injective_mulHom (Prod.upMulHom G H) (fun x y he ↦ Prod.ext ?_ ?_)
                                               /-
                                                 case refine_1.refine_1
                                                 G : Type u
                                                 H : Type v
                                                 inst✝³ : Mul G
                                                 inst✝² : Mul H
                                                 inst✝¹ : TwoUniqueProds G
                                                 inst✝ : TwoUniqueProds H
                                                 this : ∀ (b : Bool), TwoUniqueProds (I G H b)
                                                 x y : Prod G H
                                                 he : Eq ((Prod.upMulHom G H) x) ((Prod.upMulHom G H) y)
                                                 ⊢ Eq x.1 y.1
                                               -/
      (TwoUniqueProds.instForall <| I G H) <;> apply up_injective
    /-
      case refine_1.refine_1.a
      G : Type u
      H : Type v
      inst✝³ : Mul G
      inst✝² : Mul H
      inst✝¹ : TwoUniqueProds G
      inst✝ : TwoUniqueProds H
      this : ∀ (b : Bool), TwoUniqueProds (I G H b)
      x y : Prod G H
      he : Eq ((Prod.upMulHom G H) x) ((Prod.upMulHom G H) y)
      ⊢ Eq { down := x.1 } { down := y.1 }
    -/
    exacts [congr_fun he false, congr_fun he true]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      G : Type u
      H : Type v
      inst✝³ : Mul G
      inst✝² : Mul H
      inst✝¹ : TwoUniqueProds G
      inst✝ : TwoUniqueProds H
      ⊢ TwoUniqueProds (I G H Bool.false)
    -/
  · exact of_injective_mulHom (downMulHom G) down_injective ‹_›
    /-
      🎉 no goals
    -/


open MulOpposite in
@[to_additive]
theorem of_mulOpposite (h : TwoUniqueProds Gᵐᵒᵖ) : TwoUniqueProds G where
  uniqueMul_of_one_lt_card hc := by
    /-
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      hc : LT.lt 1 (HMul.hMul A✝.card B✝.card)
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A✝ B✝) p1) (Exists fun p2  …
    -/
    let f : G ↪ Gᵐᵒᵖ := ⟨op, op_injective⟩
    /-
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      hc : LT.lt 1 (HMul.hMul A✝.card B✝.card)
      f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A✝ B✝) p1) (Exists fun p2  …
    -/
    rw [← card_map f, ← card_map f, mul_comm] at hc
    /-
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
      hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A✝ B✝) p1) (Exists fun p2  …
    -/
    obtain ⟨p1, h1, p2, h2, hne, hu1, hu2⟩ := h.uniqueMul_of_one_lt_card hc
    /-
      case intro.intro.intro.intro.intro.intro
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
      hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
      p1 : Prod (MulOpposite G) (MulOpposite G)
      h1 : Membership.mem (SProd.sprod (Finset.map f B✝) (Finset.map f A✝)) p1
      p2 : Prod (MulOpposite G) (MulOpposite G)
      h2 : Membership.mem (SProd.sprod (Finset.map f B✝) (Finset.map f A✝)) p2
      hne : Ne p1 p2
      hu1 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p1.1 p1.2
      hu2 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p2.1 p2.2
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A✝ B✝) p1) (Exists fun p2  …
    -/
    simp_rw [mem_product] at h1 h2 ⊢
    /-
      case intro.intro.intro.intro.intro.intro
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
      hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
      p1 p2 : Prod (MulOpposite G) (MulOpposite G)
      hne : Ne p1 p2
      hu1 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p1.1 p1.2
      hu2 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p2.1 p2.2
      h1 : And (Membership.mem (Finset.map f B✝) p1.1) (Membership.mem (Finset.map f …
      h2 : And (Membership.mem (Finset.map f B✝) p2.1) (Membership.mem (Finset.map f …
      ⊢ Exists fun p1 => And (And (Membership.mem A✝ p1.1) (Membership.mem B✝ p1.2)) …
    -/
    refine ⟨(_, _), ⟨?_, ?_⟩, (_, _), ⟨?_, ?_⟩, ?_, hu1.of_mulOpposite, hu2.of_mulOpposite⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
      hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
      p1 p2 : Prod (MulOpposite G) (MulOpposite G)
      hne : Ne p1 p2
      hu1 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p1.1 p1.2
      hu2 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p2.1 p2.2
      h1 : And (Membership.mem (Finset.map f B✝) p1.1) (Membership.mem (Finset.map f …
      h2 : And (Membership.mem (Finset.map f B✝) p2.1) (Membership.mem (Finset.map f …
      ⊢ Membership.mem A✝ { fst := p1.2.1, snd := p1.1.1 }.1
    -/
    pick_goal 5
      /-
        case intro.intro.intro.intro.intro.intro.refine_5
        G : Type u
        inst✝ : Mul G
        h : TwoUniqueProds (MulOpposite G)
        A✝ B✝ : Finset G
        f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
        hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
        p1 p2 : Prod (MulOpposite G) (MulOpposite G)
        hne : Ne p1 p2
        hu1 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p1.1 p1.2
        hu2 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p2.1 p2.2
        h1 : And (Membership.mem (Finset.map f B✝) p1.1) (Membership.mem (Finset.map f …
        h2 : And (Membership.mem (Finset.map f B✝) p2.1) (Membership.mem (Finset.map f …
        ⊢ Ne { fst := p1.2.1, snd := p1.1.1 } { fst := p2.2.1, snd := p2.1.1 }
      -/
    · contrapose! hne; rw [Prod.ext_iff] at hne ⊢
      /-
        case intro.intro.intro.intro.intro.intro.refine_5
        G : Type u
        inst✝ : Mul G
        h : TwoUniqueProds (MulOpposite G)
        A✝ B✝ : Finset G
        f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
        hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
        p1 p2 : Prod (MulOpposite G) (MulOpposite G)
        hu1 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p1.1 p1.2
        hu2 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p2.1 p2.2
        h1 : And (Membership.mem (Finset.map f B✝) p1.1) (Membership.mem (Finset.map f …
        h2 : And (Membership.mem (Finset.map f B✝) p2.1) (Membership.mem (Finset.map f …
        hne : And (Eq { fst := p1.2.1, snd := p1.1.1 }.1 { fst := p2.2.1, snd := p2.1. …
        ⊢ And (Eq p1.1 p2.1) (Eq p1.2 p2.2)
      -/
      exact ⟨unop_injective hne.2, unop_injective hne.1⟩
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
      hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
      p1 p2 : Prod (MulOpposite G) (MulOpposite G)
      hne : Ne p1 p2
      hu1 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p1.1 p1.2
      hu2 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p2.1 p2.2
      h1 : And (Membership.mem (Finset.map f B✝) p1.1) (Membership.mem (Finset.map f …
      h2 : And (Membership.mem (Finset.map f B✝) p2.1) (Membership.mem (Finset.map f …
      ⊢ Membership.mem A✝ { fst := p1.2.1, snd := p1.1.1 }.1
    -/
    all_goals apply (mem_map' f).mp
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      G : Type u
      inst✝ : Mul G
      h : TwoUniqueProds (MulOpposite G)
      A✝ B✝ : Finset G
      f : Function.Embedding G (MulOpposite G) := { toFun := MulOpposite.op, inj' := …
      hc : LT.lt 1 (HMul.hMul (Finset.map f B✝).card (Finset.map f A✝).card)
      p1 p2 : Prod (MulOpposite G) (MulOpposite G)
      hne : Ne p1 p2
      hu1 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p1.1 p1.2
      hu2 : UniqueMul (Finset.map f B✝) (Finset.map f A✝) p2.1 p2.2
      h1 : And (Membership.mem (Finset.map f B✝) p1.1) (Membership.mem (Finset.map f …
      h2 : And (Membership.mem (Finset.map f B✝) p2.1) (Membership.mem (Finset.map f …
      ⊢ Membership.mem (Finset.map f A✝) (f { fst := p1.2.1, snd := p1.1.1 }.1)
    -/
    exacts [h1.2, h1.1, h2.2, h2.1]
    /-
      🎉 no goals
    -/


@[to_additive] instance [h : TwoUniqueProds G] : TwoUniqueProds Gᵐᵒᵖ :=
  of_mulOpposite <| (MulEquiv.opOp G).twoUniqueProds_iff.mp h

-- see Note [lower instance priority]

/-- This instance asserts that if `G` has a right-cancellative multiplication, a linear order, and
  multiplication is strictly monotone w.r.t. the second argument, then `G` has `TwoUniqueProds`. -/
@[to_additive
  "This instance asserts that if `G` has a right-cancellative addition, a linear order,
  and addition is strictly monotone w.r.t. the second argument, then `G` has `TwoUniqueSums`." ]
instance (priority := 100) of_covariant_right [IsRightCancelMul G]
    [LinearOrder G] [MulLeftStrictMono G] :
    TwoUniqueProds G where
  uniqueMul_of_one_lt_card {A B} hc := by
    /-
      G : Type u
      H : Type v
      inst✝⁴ : Mul G
      inst✝³ : Mul H
      inst✝² : IsRightCancelMul G
      inst✝¹ : LinearOrder G
      inst✝ : MulLeftStrictMono G
      A B : Finset G
      hc : LT.lt 1 (HMul.hMul A.card B.card)
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    obtain ⟨hA, hB, -⟩ := Nat.one_lt_mul_iff.mp hc
    /-
      case intro.intro
      G : Type u
      H : Type v
      inst✝⁴ : Mul G
      inst✝³ : Mul H
      inst✝² : IsRightCancelMul G
      inst✝¹ : LinearOrder G
      inst✝ : MulLeftStrictMono G
      A B : Finset G
      hc : LT.lt 1 (HMul.hMul A.card B.card)
      hA : LT.lt 0 A.card
      hB : LT.lt 0 B.card
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    rw [card_pos] at hA hB
    /-
      case intro.intro
      G : Type u
      H : Type v
      inst✝⁴ : Mul G
      inst✝³ : Mul H
      inst✝² : IsRightCancelMul G
      inst✝¹ : LinearOrder G
      inst✝ : MulLeftStrictMono G
      A B : Finset G
      hc : LT.lt 1 (HMul.hMul A.card B.card)
      hA : A.Nonempty
      hB : B.Nonempty
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    rw [← card_product] at hc
    /-
      case intro.intro
      G : Type u
      H : Type v
      inst✝⁴ : Mul G
      inst✝³ : Mul H
      inst✝² : IsRightCancelMul G
      inst✝¹ : LinearOrder G
      inst✝ : MulLeftStrictMono G
      A B : Finset G
      hc : LT.lt 1 (SProd.sprod A B).card
      hA : A.Nonempty
      hB : B.Nonempty
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    obtain ⟨a0, ha0, b0, hb0, he0⟩ := mem_mul.mp (max'_mem _ <| hA.mul hB)
    /-
      case intro.intro.intro.intro.intro.intro
      G : Type u
      H : Type v
      inst✝⁴ : Mul G
      inst✝³ : Mul H
      inst✝² : IsRightCancelMul G
      inst✝¹ : LinearOrder G
      inst✝ : MulLeftStrictMono G
      A B : Finset G
      hc : LT.lt 1 (SProd.sprod A B).card
      hA : A.Nonempty
      hB : B.Nonempty
      a0 : G
      ha0 : Membership.mem A a0
      b0 : G
      hb0 : Membership.mem B b0
      he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    obtain ⟨a1, ha1, b1, hb1, he1⟩ := mem_mul.mp (min'_mem _ <| hA.mul hB)
    have : UniqueMul A B a0 b0 := by
      intro a b ha hb he
      obtain hl | rfl | hl := lt_trichotomy b b0
      · exact ((he0 ▸ he ▸ mul_lt_mul_left' hl a).not_le <| le_max' _ _ <| mul_mem_mul ha hb0).elim
      · exact ⟨mul_right_cancel he, rfl⟩
      · exact ((he0 ▸ mul_lt_mul_left' hl a0).not_le <| le_max' _ _ <| mul_mem_mul ha0 hb).elim
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      G : Type u
      H : Type v
      inst✝⁴ : Mul G
      inst✝³ : Mul H
      inst✝² : IsRightCancelMul G
      inst✝¹ : LinearOrder G
      inst✝ : MulLeftStrictMono G
      A B : Finset G
      hc : LT.lt 1 (SProd.sprod A B).card
      hA : A.Nonempty
      hB : B.Nonempty
      a0 : G
      ha0 : Membership.mem A a0
      b0 : G
      hb0 : Membership.mem B b0
      he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
      a1 : G
      ha1 : Membership.mem A a1
      b1 : G
      hb1 : Membership.mem B b1
      he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
      this : UniqueMul A B a0 b0
      ⊢ Exists fun p1 => And (Membership.mem (SProd.sprod A B) p1) (Exists fun p2 => …
    -/
    refine ⟨_, mk_mem_product ha0 hb0, _, mk_mem_product ha1 hb1, fun he ↦ ?_, this, ?_⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        G : Type u
        H : Type v
        inst✝⁴ : Mul G
        inst✝³ : Mul H
        inst✝² : IsRightCancelMul G
        inst✝¹ : LinearOrder G
        inst✝ : MulLeftStrictMono G
        A B : Finset G
        hc : LT.lt 1 (SProd.sprod A B).card
        hA : A.Nonempty
        hB : B.Nonempty
        a0 : G
        ha0 : Membership.mem A a0
        b0 : G
        hb0 : Membership.mem B b0
        he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
        a1 : G
        ha1 : Membership.mem A a1
        b1 : G
        hb1 : Membership.mem B b1
        he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
        this : UniqueMul A B a0 b0
        he : Eq { fst := a0, snd := b0 } { fst := a1, snd := b1 }
        ⊢ False
      -/
    · rw [Prod.mk.inj_iff] at he; rw [he.1, he.2, he1] at he0
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        G : Type u
        H : Type v
        inst✝⁴ : Mul G
        inst✝³ : Mul H
        inst✝² : IsRightCancelMul G
        inst✝¹ : LinearOrder G
        inst✝ : MulLeftStrictMono G
        A B : Finset G
        hc : LT.lt 1 (SProd.sprod A B).card
        hA : A.Nonempty
        hB : B.Nonempty
        a0 : G
        ha0 : Membership.mem A a0
        b0 : G
        hb0 : Membership.mem B b0
        a1 : G
        ha1 : Membership.mem A a1
        b1 : G
        he0 : Eq ((HMul.hMul A B).min' ⋯) ((HMul.hMul A B).max' ⋯)
        hb1 : Membership.mem B b1
        he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
        this : UniqueMul A B a0 b0
        he : And (Eq a0 a1) (Eq b0 b1)
        ⊢ False
      -/
      obtain ⟨⟨a2, b2⟩, h2, hne⟩ := exists_ne_of_one_lt_card hc (a0, b0)
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1.intr …
        G : Type u
        H : Type v
        inst✝⁴ : Mul G
        inst✝³ : Mul H
        inst✝² : IsRightCancelMul G
        inst✝¹ : LinearOrder G
        inst✝ : MulLeftStrictMono G
        A B : Finset G
        hc : LT.lt 1 (SProd.sprod A B).card
        hA : A.Nonempty
        hB : B.Nonempty
        a0 : G
        ha0 : Membership.mem A a0
        b0 : G
        hb0 : Membership.mem B b0
        a1 : G
        ha1 : Membership.mem A a1
        b1 : G
        he0 : Eq ((HMul.hMul A B).min' ⋯) ((HMul.hMul A B).max' ⋯)
        hb1 : Membership.mem B b1
        he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
        this : UniqueMul A B a0 b0
        he : And (Eq a0 a1) (Eq b0 b1)
        a2 b2 : G
        h2 : Membership.mem (SProd.sprod A B) { fst := a2, snd := b2 }
        hne : Ne { fst := a2, snd := b2 } { fst := a0, snd := b0 }
        ⊢ False
      -/
      rw [mem_product] at h2
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1.intr …
        G : Type u
        H : Type v
        inst✝⁴ : Mul G
        inst✝³ : Mul H
        inst✝² : IsRightCancelMul G
        inst✝¹ : LinearOrder G
        inst✝ : MulLeftStrictMono G
        A B : Finset G
        hc : LT.lt 1 (SProd.sprod A B).card
        hA : A.Nonempty
        hB : B.Nonempty
        a0 : G
        ha0 : Membership.mem A a0
        b0 : G
        hb0 : Membership.mem B b0
        a1 : G
        ha1 : Membership.mem A a1
        b1 : G
        he0 : Eq ((HMul.hMul A B).min' ⋯) ((HMul.hMul A B).max' ⋯)
        hb1 : Membership.mem B b1
        he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
        this : UniqueMul A B a0 b0
        he : And (Eq a0 a1) (Eq b0 b1)
        a2 b2 : G
        h2 : And (Membership.mem A { fst := a2, snd := b2 }.1) (Membership.mem B { fst …
        hne : Ne { fst := a2, snd := b2 } { fst := a0, snd := b0 }
        ⊢ False
      -/
      refine (min'_lt_max' _ (mul_mem_mul ha0 hb0) (mul_mem_mul h2.1 h2.2) fun he ↦ hne ?_).ne he0
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1.intr …
        G : Type u
        H : Type v
        inst✝⁴ : Mul G
        inst✝³ : Mul H
        inst✝² : IsRightCancelMul G
        inst✝¹ : LinearOrder G
        inst✝ : MulLeftStrictMono G
        A B : Finset G
        hc : LT.lt 1 (SProd.sprod A B).card
        hA : A.Nonempty
        hB : B.Nonempty
        a0 : G
        ha0 : Membership.mem A a0
        b0 : G
        hb0 : Membership.mem B b0
        a1 : G
        ha1 : Membership.mem A a1
        b1 : G
        he0 : Eq ((HMul.hMul A B).min' ⋯) ((HMul.hMul A B).max' ⋯)
        hb1 : Membership.mem B b1
        he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
        this : UniqueMul A B a0 b0
        he✝ : And (Eq a0 a1) (Eq b0 b1)
        a2 b2 : G
        h2 : And (Membership.mem A { fst := a2, snd := b2 }.1) (Membership.mem B { fst …
        hne : Ne { fst := a2, snd := b2 } { fst := a0, snd := b0 }
        he : Eq (HMul.hMul a0 b0) (HMul.hMul { fst := a2, snd := b2 }.1 { fst := a2, s …
        ⊢ Eq { fst := a2, snd := b2 } { fst := a0, snd := b0 }
      -/
      exact Prod.ext_iff.mpr (this h2.1 h2.2 he.symm)
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        G : Type u
        H : Type v
        inst✝⁴ : Mul G
        inst✝³ : Mul H
        inst✝² : IsRightCancelMul G
        inst✝¹ : LinearOrder G
        inst✝ : MulLeftStrictMono G
        A B : Finset G
        hc : LT.lt 1 (SProd.sprod A B).card
        hA : A.Nonempty
        hB : B.Nonempty
        a0 : G
        ha0 : Membership.mem A a0
        b0 : G
        hb0 : Membership.mem B b0
        he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
        a1 : G
        ha1 : Membership.mem A a1
        b1 : G
        hb1 : Membership.mem B b1
        he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
        this : UniqueMul A B a0 b0
        ⊢ UniqueMul A B { fst := a1, snd := b1 }.1 { fst := a1, snd := b1 }.2
      -/
    · intro a b ha hb he
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        G : Type u
        H : Type v
        inst✝⁴ : Mul G
        inst✝³ : Mul H
        inst✝² : IsRightCancelMul G
        inst✝¹ : LinearOrder G
        inst✝ : MulLeftStrictMono G
        A B : Finset G
        hc : LT.lt 1 (SProd.sprod A B).card
        hA : A.Nonempty
        hB : B.Nonempty
        a0 : G
        ha0 : Membership.mem A a0
        b0 : G
        hb0 : Membership.mem B b0
        he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
        a1 : G
        ha1 : Membership.mem A a1
        b1 : G
        hb1 : Membership.mem B b1
        he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
        this : UniqueMul A B a0 b0
        a b : G
        ha : Membership.mem A a
        hb : Membership.mem B b
        he : Eq (HMul.hMul a b) (HMul.hMul { fst := a1, snd := b1 }.1 { fst := a1, snd …
        ⊢ And (Eq a { fst := a1, snd := b1 }.1) (Eq b { fst := a1, snd := b1 }.2)
      -/
      obtain hl | rfl | hl := lt_trichotomy b b1
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2.inl
          G : Type u
          H : Type v
          inst✝⁴ : Mul G
          inst✝³ : Mul H
          inst✝² : IsRightCancelMul G
          inst✝¹ : LinearOrder G
          inst✝ : MulLeftStrictMono G
          A B : Finset G
          hc : LT.lt 1 (SProd.sprod A B).card
          hA : A.Nonempty
          hB : B.Nonempty
          a0 : G
          ha0 : Membership.mem A a0
          b0 : G
          hb0 : Membership.mem B b0
          he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
          a1 : G
          ha1 : Membership.mem A a1
          b1 : G
          hb1 : Membership.mem B b1
          he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
          this : UniqueMul A B a0 b0
          a b : G
          ha : Membership.mem A a
          hb : Membership.mem B b
          he : Eq (HMul.hMul a b) (HMul.hMul { fst := a1, snd := b1 }.1 { fst := a1, snd …
          hl : LT.lt b b1
          ⊢ And (Eq a { fst := a1, snd := b1 }.1) (Eq b { fst := a1, snd := b1 }.2)
        -/
      · exact ((he1 ▸ mul_lt_mul_left' hl a1).not_le <| min'_le _ _ <| mul_mem_mul ha1 hb).elim
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2.inr. …
          G : Type u
          H : Type v
          inst✝⁴ : Mul G
          inst✝³ : Mul H
          inst✝² : IsRightCancelMul G
          inst✝¹ : LinearOrder G
          inst✝ : MulLeftStrictMono G
          A B : Finset G
          hc : LT.lt 1 (SProd.sprod A B).card
          hA : A.Nonempty
          hB : B.Nonempty
          a0 : G
          ha0 : Membership.mem A a0
          b0 : G
          hb0 : Membership.mem B b0
          he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
          a1 : G
          ha1 : Membership.mem A a1
          this : UniqueMul A B a0 b0
          a b : G
          ha : Membership.mem A a
          hb hb1 : Membership.mem B b
          he1 : Eq (HMul.hMul a1 b) ((HMul.hMul A B).min' ⋯)
          he : Eq (HMul.hMul a b) (HMul.hMul { fst := a1, snd := b }.1 { fst := a1, snd  …
          ⊢ And (Eq a { fst := a1, snd := b }.1) (Eq b { fst := a1, snd := b }.2)
        -/
      · exact ⟨mul_right_cancel he, rfl⟩
        /-
          🎉 no goals
        -/
        /-
          case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2.inr. …
          G : Type u
          H : Type v
          inst✝⁴ : Mul G
          inst✝³ : Mul H
          inst✝² : IsRightCancelMul G
          inst✝¹ : LinearOrder G
          inst✝ : MulLeftStrictMono G
          A B : Finset G
          hc : LT.lt 1 (SProd.sprod A B).card
          hA : A.Nonempty
          hB : B.Nonempty
          a0 : G
          ha0 : Membership.mem A a0
          b0 : G
          hb0 : Membership.mem B b0
          he0 : Eq (HMul.hMul a0 b0) ((HMul.hMul A B).max' ⋯)
          a1 : G
          ha1 : Membership.mem A a1
          b1 : G
          hb1 : Membership.mem B b1
          he1 : Eq (HMul.hMul a1 b1) ((HMul.hMul A B).min' ⋯)
          this : UniqueMul A B a0 b0
          a b : G
          ha : Membership.mem A a
          hb : Membership.mem B b
          he : Eq (HMul.hMul a b) (HMul.hMul { fst := a1, snd := b1 }.1 { fst := a1, snd …
          hl : LT.lt b1 b
          ⊢ And (Eq a { fst := a1, snd := b1 }.1) (Eq b { fst := a1, snd := b1 }.2)
        -/
      · exact ((he1 ▸ he ▸ mul_lt_mul_left' hl a).not_le <| min'_le _ _ <| mul_mem_mul ha hb1).elim
        /-
          🎉 no goals
        -/


open MulOpposite in
-- see Note [lower instance priority]
/-- This instance asserts that if `G` has a left-cancellative multiplication, a linear order, and
  multiplication is strictly monotone w.r.t. the first argument, then `G` has `TwoUniqueProds`. -/
@[to_additive
  "This instance asserts that if `G` has a left-cancellative addition, a linear order, and
  addition is strictly monotone w.r.t. the first argument, then `G` has `TwoUniqueSums`." ]
instance (priority := 100) of_covariant_left [IsLeftCancelMul G]
    [LinearOrder G] [MulRightStrictMono G] :
    TwoUniqueProds G :=
  let _ := LinearOrder.lift' (unop : Gᵐᵒᵖ → G) unop_injective
  let _ : MulLeftStrictMono Gᵐᵒᵖ :=
    { elim := fun _ _ _ bc ↦ mul_lt_mul_right' (α := G) bc (unop _) }
  of_mulOpposite of_covariant_right


@[deprecated (since := "2024-02-04")]
alias UniqueProds.mulHom_image_of_injective := UniqueProds.of_injective_mulHom

@[deprecated (since := "2024-02-04")]
alias UniqueSums.addHom_image_of_injective := UniqueSums.of_injective_addHom

@[deprecated (since := "2024-02-04")]
alias UniqueProds.mulHom_image_iff := MulEquiv.uniqueProds_iff

@[deprecated (since := "2024-02-04")]
alias UniqueSums.addHom_image_iff := AddEquiv.uniqueSums_iff

@[deprecated (since := "2024-02-04")]
alias TwoUniqueProds.mulHom_image_of_injective := TwoUniqueProds.of_injective_mulHom

@[deprecated (since := "2024-02-04")]
alias TwoUniqueSums.addHom_image_of_injective := TwoUniqueSums.of_injective_addHom

@[deprecated (since := "2024-02-04")]
alias TwoUniqueProds.mulHom_image_iff := MulEquiv.twoUniqueProds_iff

@[deprecated (since := "2024-02-04")]
alias TwoUniqueSums.addHom_image_iff := AddEquiv.twoUniqueSums_iff


instance {ι} (G : ι → Type*) [∀ i, AddZeroClass (G i)] [∀ i, TwoUniqueSums (G i)] :
    TwoUniqueSums (Π₀ i, G i) :=
  TwoUniqueSums.of_injective_addHom
    DFinsupp.coeFnAddMonoidHom.toAddHom DFunLike.coe_injective inferInstance


instance {ι G} [AddZeroClass G] [TwoUniqueSums G] : TwoUniqueSums (ι →₀ G) :=
  TwoUniqueSums.of_injective_addHom
    Finsupp.coeFnAddHom.toAddHom DFunLike.coe_injective inferInstance


/-- Any `FreeMonoid` has the `TwoUniqueProds` property. -/
instance FreeMonoid.instTwoUniqueProds {κ : Type*} : TwoUniqueProds (FreeMonoid κ) :=
  .of_mulHom ⟨Multiplicative.ofAdd ∘ List.length, fun _ _ ↦ congr_arg _ (List.length_append _ _)⟩
    (fun _ _ _ _ h h' ↦ List.append_inj h <| Equiv.injective Multiplicative.ofAdd h'.1)


/-- Any `FreeAddMonoid` has the `TwoUniqueSums` property. -/
instance FreeAddMonoid.instTwoUniqueSums {κ : Type*} : TwoUniqueSums (FreeAddMonoid κ) :=
  .of_addHom ⟨_, List.length_append⟩ (fun _ _ _ _ h h' ↦ List.append_inj h h'.1)

