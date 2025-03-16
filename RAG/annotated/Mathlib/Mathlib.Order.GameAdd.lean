/-- `Prod.GameAdd rα rβ x y` means that `x` can be reached from `y` by decreasing either entry with
  respect to the relations `rα` and `rβ`.

  It is so called, as it models game addition within combinatorial game theory. If `rα a₁ a₂` means
  that `a₂ ⟶ a₁` is a valid move in game `α`, and `rβ b₁ b₂` means that `b₂ ⟶ b₁` is a valid move
  in game `β`, then `GameAdd rα rβ` specifies the valid moves in the juxtaposition of `α` and `β`:
  the player is free to choose one of the games and make a move in it, while leaving the other game
  unchanged.

  See `Sym2.GameAdd` for the unordered pair analog. -/

inductive GameAdd : α × β → α × β → Prop
  | fst {a₁ a₂ b} : rα a₁ a₂ → GameAdd (a₁, b) (a₂, b)
  | snd {a b₁ b₂} : rβ b₁ b₂ → GameAdd (a, b₁) (a, b₂)


theorem gameAdd_iff {rα rβ} {x y : α × β} :
    GameAdd rα rβ x y ↔ rα x.1 y.1 ∧ x.2 = y.2 ∨ rβ x.2 y.2 ∧ x.1 = y.1 := by
  /-
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    x y : Prod α β
    ⊢ Iff (Prod.GameAdd rα rβ x y) (Or (And (rα x.1 y.1) (Eq x.2 y.2)) (And (rβ x. …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      rα : α → α → Prop
      rβ : β → β → Prop
      x y : Prod α β
      ⊢ Prod.GameAdd rα rβ x y → Or (And (rα x.1 y.1) (Eq x.2 y.2)) (And (rβ x.2 y.2 …
    -/
  · rintro (@⟨a₁, a₂, b, h⟩ | @⟨a, b₁, b₂, h⟩)
    /-
      case mp.fst
      α : Type u_1
      β : Type u_2
      rα : α → α → Prop
      rβ : β → β → Prop
      a₁ a₂ : α
      b : β
      h : rα a₁ a₂
      ⊢ Or (And (rα { fst := a₁, snd := b }.1 { fst := a₂, snd := b }.1) (Eq { fst : …
    -/
    exacts [Or.inl ⟨h, rfl⟩, Or.inr ⟨h, rfl⟩]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      rα : α → α → Prop
      rβ : β → β → Prop
      x y : Prod α β
      ⊢ Or (And (rα x.1 y.1) (Eq x.2 y.2)) (And (rβ x.2 y.2) (Eq x.1 y.1)) → Prod.Ga …
    -/
  · revert x y
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      rα : α → α → Prop
      rβ : β → β → Prop
      ⊢ ∀ {x y : Prod α β}, Or (And (rα x.1 y.1) (Eq x.2 y.2)) (And (rβ x.2 y.2) (Eq …
    -/
    rintro ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ (⟨h, rfl : b₁ = b₂⟩ | ⟨h, rfl : a₁ = a₂⟩)
    /-
      case mpr.mk.mk.inl.intro
      α : Type u_1
      β : Type u_2
      rα : α → α → Prop
      rβ : β → β → Prop
      a₁ : α
      b₁ : β
      a₂ : α
      h : rα { fst := a₁, snd := b₁ }.1 { fst := a₂, snd := b₁ }.1
      ⊢ Prod.GameAdd rα rβ { fst := a₁, snd := b₁ } { fst := a₂, snd := b₁ }
    -/
    exacts [GameAdd.fst h, GameAdd.snd h]
    /-
      🎉 no goals
    -/


theorem gameAdd_mk_iff {rα rβ} {a₁ a₂ : α} {b₁ b₂ : β} :
    GameAdd rα rβ (a₁, b₁) (a₂, b₂) ↔ rα a₁ a₂ ∧ b₁ = b₂ ∨ rβ b₁ b₂ ∧ a₁ = a₂ :=
  gameAdd_iff


@[simp]
theorem gameAdd_swap_swap : ∀ a b : α × β, GameAdd rβ rα a.swap b.swap ↔ GameAdd rα rβ a b :=
                              /-
                                α : Type u_1
                                β : Type u_2
                                rα : α → α → Prop
                                rβ : β → β → Prop
                                x✝¹ x✝ : Prod α β
                                a₁ : α
                                b₁ : β
                                a₂ : α
                                b₂ : β
                                ⊢ Iff (Prod.GameAdd rβ rα { fst := a₁, snd := b₁ }.swap { fst := a₂, snd := b₂ …
                              -/
  fun ⟨a₁, b₁⟩ ⟨a₂, b₂⟩ => by rw [Prod.swap, Prod.swap, gameAdd_mk_iff, gameAdd_mk_iff, or_comm]
                              /-
                                🎉 no goals
                              -/


theorem gameAdd_swap_swap_mk (a₁ a₂ : α) (b₁ b₂ : β) :
    GameAdd rα rβ (a₁, b₁) (a₂, b₂) ↔ GameAdd rβ rα (b₁, a₁) (b₂, a₂) :=
  gameAdd_swap_swap rβ rα (b₁, a₁) (b₂, a₂)


/-- `Prod.GameAdd` is a subrelation of `Prod.Lex`. -/
theorem gameAdd_le_lex : GameAdd rα rβ ≤ Prod.Lex rα rβ := fun _ _ h =>
  h.rec (Prod.Lex.left _ _) (Prod.Lex.right _)


/-- `Prod.RProd` is a subrelation of the transitive closure of `Prod.GameAdd`. -/
theorem rprod_le_transGen_gameAdd : RProd rα rβ ≤ Relation.TransGen (GameAdd rα rβ)
  | _, _, h => h.rec (by
      /-
        α : Type u_1
        β : Type u_2
        rα : α → α → Prop
        rβ : β → β → Prop
        x✝¹ x✝ : Prod α β
        h : Prod.RProd rα rβ x✝¹ x✝
        ⊢ ∀ {a₁ : α} {b₁ : β} {a₂ : α} {b₂ : β}, rα a₁ a₂ → rβ b₁ b₂ → Relation.TransG …
      -/
      intro _ _ _ _ hα hβ
      /-
        α : Type u_1
        β : Type u_2
        rα : α → α → Prop
        rβ : β → β → Prop
        x✝¹ x✝ : Prod α β
        h : Prod.RProd rα rβ x✝¹ x✝
        a₁✝ : α
        b₁✝ : β
        a₂✝ : α
        b₂✝ : β
        hα : rα a₁✝ a₂✝
        hβ : rβ b₁✝ b₂✝
        ⊢ Relation.TransGen (Prod.GameAdd rα rβ) { fst := a₁✝, snd := b₁✝ } { fst := a …
      -/
      exact Relation.TransGen.tail (Relation.TransGen.single <| GameAdd.fst hα) (GameAdd.snd hβ))
      /-
        🎉 no goals
      -/


/-- If `a` is accessible under `rα` and `b` is accessible under `rβ`, then `(a, b)` is
  accessible under `Prod.GameAdd rα rβ`. Notice that `Prod.lexAccessible` requires the
  stronger condition `∀ b, Acc rβ b`. -/
theorem Acc.prod_gameAdd (ha : Acc rα a) (hb : Acc rβ b) :
    Acc (Prod.GameAdd rα rβ) (a, b) := by
  /-
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    a : α
    b : β
    ha : Acc rα a
    hb : Acc rβ b
    ⊢ Acc (Prod.GameAdd rα rβ) { fst := a, snd := b }
  -/
  induction' ha with a _ iha generalizing b
  /-
    case intro
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : β}, Acc rβ b → Acc (Prod.GameAdd rα rβ) { fst …
    b : β
    hb : Acc rβ b
    ⊢ Acc (Prod.GameAdd rα rβ) { fst := a, snd := b }
  -/
  induction' hb with b hb ihb
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : β}, Acc rβ b → Acc (Prod.GameAdd rα rβ) { fst …
    b✝ b : β
    hb : ∀ (y : β), rβ y b → Acc rβ y
    ihb : ∀ (y : β), rβ y b → Acc (Prod.GameAdd rα rβ) { fst := a, snd := y }
    ⊢ Acc (Prod.GameAdd rα rβ) { fst := a, snd := b }
  -/
  refine Acc.intro _ fun h => ?_
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : β}, Acc rβ b → Acc (Prod.GameAdd rα rβ) { fst …
    b✝ b : β
    hb : ∀ (y : β), rβ y b → Acc rβ y
    ihb : ∀ (y : β), rβ y b → Acc (Prod.GameAdd rα rβ) { fst := a, snd := y }
    h : Prod α β
    ⊢ Prod.GameAdd rα rβ h { fst := a, snd := b } → Acc (Prod.GameAdd rα rβ) h
  -/
  rintro (⟨ra⟩ | ⟨rb⟩)
  /-
    case intro.intro.fst
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : β}, Acc rβ b → Acc (Prod.GameAdd rα rβ) { fst …
    b✝ b : β
    hb : ∀ (y : β), rβ y b → Acc rβ y
    ihb : ∀ (y : β), rβ y b → Acc (Prod.GameAdd rα rβ) { fst := a, snd := y }
    a₁✝ : α
    ra : rα a₁✝ a
    ⊢ Acc (Prod.GameAdd rα rβ) { fst := a₁✝, snd := b }
  -/
  exacts [iha _ ra (Acc.intro b hb), ihb _ rb]
  /-
    🎉 no goals
  -/


/-- The `Prod.GameAdd` relation on well-founded inputs is well-founded.

  In particular, the sum of two well-founded games is well-founded. -/
theorem WellFounded.prod_gameAdd (hα : WellFounded rα) (hβ : WellFounded rβ) :
    WellFounded (Prod.GameAdd rα rβ) :=
  ⟨fun ⟨a, b⟩ => (hα.apply a).prod_gameAdd (hβ.apply b)⟩


/-- Recursion on the well-founded `Prod.GameAdd` relation.
  Note that it's strictly more general to recurse on the lexicographic order instead. -/
def GameAdd.fix {C : α → β → Sort*} (hα : WellFounded rα) (hβ : WellFounded rβ)
    (IH : ∀ a₁ b₁, (∀ a₂ b₂, GameAdd rα rβ (a₂, b₂) (a₁, b₁) → C a₂ b₂) → C a₁ b₁) (a : α) (b : β) :
    C a b :=
  @WellFounded.fix (α × β) (fun x => C x.1 x.2) _ (hα.prod_gameAdd hβ)
    (fun ⟨x₁, x₂⟩ IH' => IH x₁ x₂ fun a' b' => IH' ⟨a', b'⟩) ⟨a, b⟩


theorem GameAdd.fix_eq {C : α → β → Sort*} (hα : WellFounded rα) (hβ : WellFounded rβ)
    (IH : ∀ a₁ b₁, (∀ a₂ b₂, GameAdd rα rβ (a₂, b₂) (a₁, b₁) → C a₂ b₂) → C a₁ b₁) (a : α) (b : β) :
    GameAdd.fix hα hβ IH a b = IH a b fun a' b' _ => GameAdd.fix hα hβ IH a' b' :=
  WellFounded.fix_eq _ _ _


/-- Induction on the well-founded `Prod.GameAdd` relation.
  Note that it's strictly more general to induct on the lexicographic order instead. -/
theorem GameAdd.induction {C : α → β → Prop} :
    WellFounded rα →
      WellFounded rβ →
        (∀ a₁ b₁, (∀ a₂ b₂, GameAdd rα rβ (a₂, b₂) (a₁, b₁) → C a₂ b₂) → C a₁ b₁) → ∀ a b, C a b :=
  GameAdd.fix


/-- `Sym2.GameAdd rα x y` means that `x` can be reached from `y` by decreasing either entry with
  respect to the relation `rα`.

  See `Prod.GameAdd` for the ordered pair analog. -/
def GameAdd (rα : α → α → Prop) : Sym2 α → Sym2 α → Prop :=
  Sym2.lift₂
    ⟨fun a₁ b₁ a₂ b₂ => Prod.GameAdd rα rα (a₁, b₁) (a₂, b₂) ∨ Prod.GameAdd rα rα (b₁, a₁) (a₂, b₂),
      fun a₁ b₁ a₂ b₂ => by
        /-
          α : Type u_1
          β : Type u_2
          rα✝ : α → α → Prop
          rβ : β → β → Prop
          a : α
          b : β
          rα : α → α → Prop
          a₁ b₁ a₂ b₂ : α
          ⊢ And (Eq ((fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd := b₁ } …
        -/
        dsimp
        /-
          α : Type u_1
          β : Type u_2
          rα✝ : α → α → Prop
          rβ : β → β → Prop
          a : α
          b : β
          rα : α → α → Prop
          a₁ b₁ a₂ b₂ : α
          ⊢ And (Eq (Or (Prod.GameAdd rα rα { fst := a₁, snd := b₁ } { fst := a₂, snd := …
        -/
        rw [Prod.gameAdd_swap_swap_mk _ _ b₁ b₂ a₁ a₂, Prod.gameAdd_swap_swap_mk _ _ a₁ b₂ b₁ a₂]
        /-
          α : Type u_1
          β : Type u_2
          rα✝ : α → α → Prop
          rβ : β → β → Prop
          a : α
          b : β
          rα : α → α → Prop
          a₁ b₁ a₂ b₂ : α
          ⊢ And (Eq (Or (Prod.GameAdd rα rα { fst := a₁, snd := b₁ } { fst := a₂, snd := …
        -/
        simp [or_comm]⟩
        /-
          🎉 no goals
        -/


theorem gameAdd_iff : ∀ {x y : α × α},
    GameAdd rα (Sym2.mk x) (Sym2.mk y) ↔ Prod.GameAdd rα rα x y ∨ Prod.GameAdd rα rα x.swap y := by
  /-
    α : Type u_1
    rα : α → α → Prop
    ⊢ ∀ {x y : Prod α α}, Iff (Sym2.GameAdd rα (Sym2.mk x) (Sym2.mk y)) (Or (Prod. …
  -/
  rintro ⟨_, _⟩ ⟨_, _⟩
  /-
    case mk.mk
    α : Type u_1
    rα : α → α → Prop
    fst✝¹ snd✝¹ fst✝ snd✝ : α
    ⊢ Iff (Sym2.GameAdd rα (Sym2.mk { fst := fst✝¹, snd := snd✝¹ }) (Sym2.mk { fst …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem gameAdd_mk'_iff {a₁ a₂ b₁ b₂ : α} :
    GameAdd rα s(a₁, b₁) s(a₂, b₂) ↔
      Prod.GameAdd rα rα (a₁, b₁) (a₂, b₂) ∨ Prod.GameAdd rα rα (b₁, a₁) (a₂, b₂) :=
  Iff.rfl


theorem _root_.Prod.GameAdd.to_sym2 {a₁ a₂ b₁ b₂ : α} (h : Prod.GameAdd rα rα (a₁, b₁) (a₂, b₂)) :
    Sym2.GameAdd rα s(a₁, b₁) s(a₂, b₂) :=
  gameAdd_mk'_iff.2 <| Or.inl <| h


theorem GameAdd.fst {a₁ a₂ b : α} (h : rα a₁ a₂) : GameAdd rα s(a₁, b) s(a₂, b) :=
  (Prod.GameAdd.fst h).to_sym2


theorem GameAdd.snd {a b₁ b₂ : α} (h : rα b₁ b₂) : GameAdd rα s(a, b₁) s(a, b₂) :=
  (Prod.GameAdd.snd h).to_sym2


theorem GameAdd.fst_snd {a₁ a₂ b : α} (h : rα a₁ a₂) : GameAdd rα s(a₁, b) s(b, a₂) := by
  /-
    α : Type u_1
    rα : α → α → Prop
    a₁ a₂ b : α
    h : rα a₁ a₂
    ⊢ Sym2.GameAdd rα (Sym2.mk { fst := a₁, snd := b }) (Sym2.mk { fst := b, snd : …
  -/
  rw [Sym2.eq_swap]
  /-
    α : Type u_1
    rα : α → α → Prop
    a₁ a₂ b : α
    h : rα a₁ a₂
    ⊢ Sym2.GameAdd rα (Sym2.mk { fst := b, snd := a₁ }) (Sym2.mk { fst := b, snd : …
  -/
  exact GameAdd.snd h
  /-
    🎉 no goals
  -/


theorem GameAdd.snd_fst {a₁ a₂ b : α} (h : rα a₁ a₂) : GameAdd rα s(b, a₁) s(a₂, b) := by
  /-
    α : Type u_1
    rα : α → α → Prop
    a₁ a₂ b : α
    h : rα a₁ a₂
    ⊢ Sym2.GameAdd rα (Sym2.mk { fst := b, snd := a₁ }) (Sym2.mk { fst := a₂, snd  …
  -/
  rw [Sym2.eq_swap]
  /-
    α : Type u_1
    rα : α → α → Prop
    a₁ a₂ b : α
    h : rα a₁ a₂
    ⊢ Sym2.GameAdd rα (Sym2.mk { fst := a₁, snd := b }) (Sym2.mk { fst := a₂, snd  …
  -/
  exact GameAdd.fst h
  /-
    🎉 no goals
  -/


theorem Acc.sym2_gameAdd {a b} (ha : Acc rα a) (hb : Acc rα b) :
    Acc (Sym2.GameAdd rα) s(a, b) := by
  /-
    α : Type u_1
    rα : α → α → Prop
    a b : α
    ha : Acc rα a
    hb : Acc rα b
    ⊢ Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := b })
  -/
  induction' ha with a _ iha generalizing b
  /-
    case intro
    α : Type u_1
    rα : α → α → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
    b : α
    hb : Acc rα b
    ⊢ Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := b })
  -/
  induction' hb with b hb ihb
  /-
    case intro.intro
    α : Type u_1
    rα : α → α → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
    b✝ b : α
    hb : ∀ (y : α), rα y b → Acc rα y
    ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
    ⊢ Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := b })
  -/
  refine Acc.intro _ fun s => ?_
  /-
    case intro.intro
    α : Type u_1
    rα : α → α → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
    b✝ b : α
    hb : ∀ (y : α), rα y b → Acc rα y
    ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
    s : Sym2 α
    ⊢ Sym2.GameAdd rα s (Sym2.mk { fst := a, snd := b }) → Acc (Sym2.GameAdd rα) s
  -/
  induction' s with c d
  /-
    case intro.intro.h
    α : Type u_1
    rα : α → α → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
    b✝ b : α
    hb : ∀ (y : α), rα y b → Acc rα y
    ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
    c d : α
    ⊢ Sym2.GameAdd rα (Sym2.mk { fst := c, snd := d }) (Sym2.mk { fst := a, snd := …
  -/
  rw [Sym2.GameAdd]
  /-
    case intro.intro.h
    α : Type u_1
    rα : α → α → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
    b✝ b : α
    hb : ∀ (y : α), rα y b → Acc rα y
    ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
    c d : α
    ⊢ Sym2.lift₂ ⟨fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd := b₁ …
  -/
  dsimp
  /-
    case intro.intro.h
    α : Type u_1
    rα : α → α → Prop
    a✝ a : α
    h✝ : ∀ (y : α), rα y a → Acc rα y
    iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
    b✝ b : α
    hb : ∀ (y : α), rα y b → Acc rα y
    ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
    c d : α
    ⊢ Or (Prod.GameAdd rα rα { fst := c, snd := d } { fst := a, snd := b }) (Prod. …
  -/
  rintro ((rc | rd) | (rd | rc))
    /-
      case intro.intro.h.inl.fst
      α : Type u_1
      rα : α → α → Prop
      a✝ a : α
      h✝ : ∀ (y : α), rα y a → Acc rα y
      iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
      b✝ b : α
      hb : ∀ (y : α), rα y b → Acc rα y
      ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
      c : α
      rc : rα c a
      ⊢ Acc (Sym2.lift₂ ⟨fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd  …
    -/
  · exact iha c rc ⟨b, hb⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.h.inl.snd
      α : Type u_1
      rα : α → α → Prop
      a✝ a : α
      h✝ : ∀ (y : α), rα y a → Acc rα y
      iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
      b✝ b : α
      hb : ∀ (y : α), rα y b → Acc rα y
      ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
      d : α
      rd : rα d b
      ⊢ Acc (Sym2.lift₂ ⟨fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd  …
    -/
  · exact ihb d rd
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.h.inr.fst
      α : Type u_1
      rα : α → α → Prop
      a✝ a : α
      h✝ : ∀ (y : α), rα y a → Acc rα y
      iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
      b✝ b : α
      hb : ∀ (y : α), rα y b → Acc rα y
      ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
      d : α
      rd : rα d a
      ⊢ Acc (Sym2.lift₂ ⟨fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd  …
    -/
  · rw [Sym2.eq_swap]
    /-
      case intro.intro.h.inr.fst
      α : Type u_1
      rα : α → α → Prop
      a✝ a : α
      h✝ : ∀ (y : α), rα y a → Acc rα y
      iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
      b✝ b : α
      hb : ∀ (y : α), rα y b → Acc rα y
      ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
      d : α
      rd : rα d a
      ⊢ Acc (Sym2.lift₂ ⟨fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd  …
    -/
    exact iha d rd ⟨b, hb⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.h.inr.snd
      α : Type u_1
      rα : α → α → Prop
      a✝ a : α
      h✝ : ∀ (y : α), rα y a → Acc rα y
      iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
      b✝ b : α
      hb : ∀ (y : α), rα y b → Acc rα y
      ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
      c : α
      rc : rα c b
      ⊢ Acc (Sym2.lift₂ ⟨fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd  …
    -/
  · rw [Sym2.eq_swap]
    /-
      case intro.intro.h.inr.snd
      α : Type u_1
      rα : α → α → Prop
      a✝ a : α
      h✝ : ∀ (y : α), rα y a → Acc rα y
      iha : ∀ (y : α), rα y a → ∀ {b : α}, Acc rα b → Acc (Sym2.GameAdd rα) (Sym2.mk …
      b✝ b : α
      hb : ∀ (y : α), rα y b → Acc rα y
      ihb : ∀ (y : α), rα y b → Acc (Sym2.GameAdd rα) (Sym2.mk { fst := a, snd := y })
      c : α
      rc : rα c b
      ⊢ Acc (Sym2.lift₂ ⟨fun a₁ b₁ a₂ b₂ => Or (Prod.GameAdd rα rα { fst := a₁, snd  …
    -/
    exact ihb c rc
    /-
      🎉 no goals
    -/


/-- The `Sym2.GameAdd` relation on well-founded inputs is well-founded. -/
theorem WellFounded.sym2_gameAdd (h : WellFounded rα) : WellFounded (Sym2.GameAdd rα) :=
  ⟨fun i => Sym2.inductionOn i fun x y => (h.apply x).sym2_gameAdd (h.apply y)⟩


/-- Recursion on the well-founded `Sym2.GameAdd` relation. -/
def GameAdd.fix {C : α → α → Sort*} (hr : WellFounded rα)
    (IH : ∀ a₁ b₁, (∀ a₂ b₂, Sym2.GameAdd rα s(a₂, b₂) s(a₁, b₁) → C a₂ b₂) → C a₁ b₁) (a b : α) :
    C a b := by
  -- Porting note: this was refactored for https://github.com/leanprover-community/mathlib4/pull/3414 (reenableeta), and could perhaps be cleaned up.
  /-
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    a✝ : α
    b✝ : β
    C : α → α → Sort u_3
    hr : WellFounded rα
    IH : (a₁ b₁ : α) → ((a₂ b₂ : α) → Sym2.GameAdd rα (Sym2.mk { fst := a₂, snd := …
    a b : α
    ⊢ C a b
  -/
  have := hr.sym2_gameAdd
  /-
    α : Type u_1
    β : Type u_2
    rα : α → α → Prop
    rβ : β → β → Prop
    a✝ : α
    b✝ : β
    C : α → α → Sort u_3
    hr : WellFounded rα
    IH : (a₁ b₁ : α) → ((a₂ b₂ : α) → Sym2.GameAdd rα (Sym2.mk { fst := a₂, snd := …
    a b : α
    this : WellFounded (Sym2.GameAdd rα)
    ⊢ C a b
  -/
  dsimp only [GameAdd, lift₂, DFunLike.coe, EquivLike.coe] at this
  exact @WellFounded.fix (α × α) (fun x => C x.1 x.2) _ this.of_quotient_lift₂
    (fun ⟨x₁, x₂⟩ IH' => IH x₁ x₂ fun a' b' => IH' ⟨a', b'⟩) (a, b)


theorem GameAdd.fix_eq {C : α → α → Sort*} (hr : WellFounded rα)
    (IH : ∀ a₁ b₁, (∀ a₂ b₂, Sym2.GameAdd rα s(a₂, b₂) s(a₁, b₁) → C a₂ b₂) → C a₁ b₁) (a b : α) :
    GameAdd.fix hr IH a b = IH a b fun a' b' _ => GameAdd.fix hr IH a' b' := by
  -- Porting note: this was refactored for https://github.com/leanprover-community/mathlib4/pull/3414 (reenableeta), and could perhaps be cleaned up.
  /-
    α : Type u_1
    rα : α → α → Prop
    C : α → α → Sort u_3
    hr : WellFounded rα
    IH : (a₁ b₁ : α) → ((a₂ b₂ : α) → Sym2.GameAdd rα (Sym2.mk { fst := a₂, snd := …
    a b : α
    ⊢ Eq (Sym2.GameAdd.fix hr IH a b) (IH a b fun a' b' x => Sym2.GameAdd.fix hr I …
  -/
  dsimp [GameAdd.fix]
  /-
    α : Type u_1
    rα : α → α → Prop
    C : α → α → Sort u_3
    hr : WellFounded rα
    IH : (a₁ b₁ : α) → ((a₂ b₂ : α) → Sym2.GameAdd rα (Sym2.mk { fst := a₂, snd := …
    a b : α
    ⊢ Eq (⋯.fix (fun x IH' => IH x.1 x.2 fun a' b' => IH' { fst := a', snd := b' } …
  -/
  exact WellFounded.fix_eq _ _ _
  /-
    🎉 no goals
  -/


/-- Induction on the well-founded `Sym2.GameAdd` relation. -/
theorem GameAdd.induction {C : α → α → Prop} :
    WellFounded rα →
      (∀ a₁ b₁, (∀ a₂ b₂, Sym2.GameAdd rα s(a₂, b₂) s(a₁, b₁) → C a₂ b₂) → C a₁ b₁) →
        ∀ a b, C a b :=
  GameAdd.fix


