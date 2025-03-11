attribute [simp] cast_heq


/-- An identity function with its main argument implicit. This will be printed as `hidden` even
if it is applied to a large term, so it can be used for elision,
as done in the `elide` and `unelide` tactics. -/
abbrev hidden {α : Sort*} {a : α} := a


instance (priority := 10) decidableEq_of_subsingleton [Subsingleton α] : DecidableEq α :=
  fun a b ↦ isTrue (Subsingleton.elim a b)


instance [Subsingleton α] (p : α → Prop) : Subsingleton (Subtype p) :=
                          /-
                            α : Sort u_1
                            inst✝ : Subsingleton α
                            p : α → Prop
                            x✝¹ x✝ : Subtype p
                            x : α
                            property✝¹ : p x
                            y : α
                            property✝ : p y
                            ⊢ Eq ⟨x, property✝¹⟩ ⟨y, property✝⟩
                          -/
  ⟨fun ⟨x, _⟩ ⟨y, _⟩ ↦ by cases Subsingleton.elim x y; rfl⟩
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem congr_heq {α β γ : Sort _} {f : α → γ} {g : β → γ} {x : α} {y : β}
    (h₁ : HEq f g) (h₂ : HEq x y) : f x = g y := by
  /-
    α β : Sort u_2
    γ : Sort u_3
    f : α → γ
    g : β → γ
    x : α
    y : β
    h₁ : HEq f g
    h₂ : HEq x y
    ⊢ Eq (f x) (g y)
  -/
  cases h₂; cases h₁; rfl
                      /-
                        🎉 no goals
                      -/


theorem congr_arg_heq {β : α → Sort*} (f : ∀ a, β a) :
    ∀ {a₁ a₂ : α}, a₁ = a₂ → HEq (f a₁) (f a₂)
  | _, _, rfl => HEq.rfl


@[simp] theorem eq_iff_eq_cancel_left {b c : α} : (∀ {a}, a = b ↔ a = c) ↔ b = c :=
              /-
                α : Sort u_1
                b c : α
                h : ∀ {a : α}, Iff (Eq a b) (Eq a c)
                ⊢ Eq b c
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [← h], fun h a ↦ by rw [h]⟩
                                     /-
                                       🎉 no goals
                                     -/


@[simp] theorem eq_iff_eq_cancel_right {a b : α} : (∀ {c}, a = c ↔ b = c) ↔ a = b :=
              /-
                α : Sort u_1
                a b : α
                h : ∀ {c : α}, Iff (Eq a c) (Eq b c)
                ⊢ Eq a b
              -/
              /-
                🎉 no goals
              -/
  ⟨fun h ↦ by rw [h], fun h a ↦ by rw [h]⟩
                                   /-
                                     🎉 no goals
                                   -/


lemma ne_and_eq_iff_right {a b c : α} (h : b ≠ c) : a ≠ b ∧ a = c ↔ a = c :=
  and_iff_right_of_imp (fun h2 => h2.symm ▸ h.symm)


/-- Wrapper for adding elementary propositions to the type class systems.
Warning: this can easily be abused. See the rest of this docstring for details.

Certain propositions should not be treated as a class globally,
but sometimes it is very convenient to be able to use the type class system
in specific circumstances.

For example, `ZMod p` is a field if and only if `p` is a prime number.
In order to be able to find this field instance automatically by type class search,
we have to turn `p.prime` into an instance implicit assumption.

On the other hand, making `Nat.prime` a class would require a major refactoring of the library,
and it is questionable whether making `Nat.prime` a class is desirable at all.
The compromise is to add the assumption `[Fact p.prime]` to `ZMod.field`.

In particular, this class is not intended for turning the type class system
into an automated theorem prover for first order logic. -/
class Fact (p : Prop) : Prop where
  /-- `Fact.out` contains the unwrapped witness for the fact represented by the instance of
  `Fact p`. -/
  out : p


theorem Fact.elim {p : Prop} (h : Fact p) : p := h.1

theorem fact_iff {p : Prop} : Fact p ↔ p := ⟨fun h ↦ h.1, fun h ↦ ⟨h⟩⟩


instance {p : Prop} [Decidable p] : Decidable (Fact p) :=
  decidable_of_iff _ fact_iff.symm


/-- Swaps two pairs of arguments to a function. -/
abbrev Function.swap₂ {ι₁ ι₂ : Sort*} {κ₁ : ι₁ → Sort*} {κ₂ : ι₂ → Sort*}
    {φ : ∀ i₁, κ₁ i₁ → ∀ i₂, κ₂ i₂ → Sort*} (f : ∀ i₁ j₁ i₂ j₂, φ i₁ j₁ i₂ j₂)
    (i₂ j₂ i₁ j₁) : φ i₁ j₁ i₂ j₂ := f i₁ j₁ i₂ j₂

-- Porting note: these don't work as intended any more
-- /-- If `x : α . tac_name` then `x.out : α`. These are definitionally equal, but this can
-- nevertheless be useful for various reasons, e.g. to apply further projection notation or in an
-- argument to `simp`. -/
-- def autoParam'.out {α : Sort*} {n : Name} (x : autoParam' α n) : α := x

-- /-- If `x : α := d` then `x.out : α`. These are definitionally equal, but this can
-- nevertheless be useful for various reasons, e.g. to apply further projection notation or in an
-- argument to `simp`. -/
-- def optParam.out {α : Sort*} {d : α} (x : α := d) : α := x


alias Iff.imp := imp_congr

-- This is a duplicate of `Classical.imp_iff_right_iff`. Deprecate?

theorem imp_iff_right_iff {a b : Prop} : (a → b ↔ b) ↔ a ∨ b := Decidable.imp_iff_right_iff

-- This is a duplicate of `Classical.and_or_imp`. Deprecate?

theorem and_or_imp {a b c : Prop} : a ∧ b ∨ (a → c) ↔ a → b ∨ c := Decidable.and_or_imp


/-- Provide modus tollens (`mt`) as dot notation for implications. -/
protected theorem Function.mt {a b : Prop} : (a → b) → ¬b → ¬a := mt


alias dec_em := Decidable.em


theorem dec_em' (p : Prop) [Decidable p] : ¬p ∨ p := (dec_em p).symm


alias em := Classical.em


theorem em' (p : Prop) : ¬p ∨ p := (em p).symm


theorem or_not {p : Prop} : p ∨ ¬p := em _


theorem Decidable.eq_or_ne {α : Sort*} (x y : α) [Decidable (x = y)] : x = y ∨ x ≠ y :=
  dec_em <| x = y


theorem Decidable.ne_or_eq {α : Sort*} (x y : α) [Decidable (x = y)] : x ≠ y ∨ x = y :=
  dec_em' <| x = y


theorem eq_or_ne {α : Sort*} (x y : α) : x = y ∨ x ≠ y := em <| x = y


theorem ne_or_eq {α : Sort*} (x y : α) : x ≠ y ∨ x = y := em' <| x = y


theorem by_contradiction {p : Prop} : (¬p → False) → p := Decidable.byContradiction


theorem by_cases {p q : Prop} (hpq : p → q) (hnpq : ¬p → q) : q :=
if hp : p then hpq hp else hnpq hp


alias by_contra := by_contradiction


attribute [simp] not_not


theorem of_not_not {a : Prop} : ¬¬a → a := by_contra


theorem not_ne_iff {α : Sort*} {a b : α} : ¬a ≠ b ↔ a = b := not_not


theorem of_not_imp : ¬(a → b) → a := Decidable.of_not_imp


alias Not.decidable_imp_symm := Decidable.not_imp_symm


theorem Not.imp_symm : (¬a → b) → ¬b → a := Not.decidable_imp_symm


theorem not_imp_comm : ¬a → b ↔ ¬b → a := Decidable.not_imp_comm


@[simp] theorem not_imp_self : ¬a → a ↔ a := Decidable.not_imp_self


theorem Imp.swap {a b : Sort*} {c : Prop} : a → b → c ↔ b → a → c :=
  ⟨fun h x y ↦ h y x, fun h x y ↦ h y x⟩


alias Iff.not := not_congr


theorem Iff.not_left (h : a ↔ ¬b) : ¬a ↔ b := h.not.trans not_not


theorem Iff.not_right (h : ¬a ↔ b) : a ↔ ¬b := not_not.symm.trans h.not


protected lemma Iff.ne {α β : Sort*} {a b : α} {c d : β} : (a = b ↔ c = d) → (a ≠ b ↔ c ≠ d) :=
  Iff.not


lemma Iff.ne_left {α β : Sort*} {a b : α} {c d : β} : (a = b ↔ c ≠ d) → (a ≠ b ↔ c = d) :=
  Iff.not_left


lemma Iff.ne_right {α β : Sort*} {a b : α} {c d : β} : (a ≠ b ↔ c = d) → (a = b ↔ c ≠ d) :=
  Iff.not_right


/-- `Xor' a b` is the exclusive-or of propositions. -/
def Xor' (a b : Prop) := (a ∧ ¬b) ∨ (b ∧ ¬a)


instance [Decidable a] [Decidable b] : Decidable (Xor' a b) := inferInstanceAs (Decidable (Or ..))


@[simp] theorem xor_true : Xor' True = Not := by
  /-
    ⊢ Eq (Xor' True) Not
  -/
  simp (config := { unfoldPartialApp := true }) [Xor']
  /-
    🎉 no goals
  -/


                                                  /-
                                                    ⊢ Eq (Xor' False) id
                                                  -/
@[simp] theorem xor_false : Xor' False = id := by ext; simp [Xor']
                                                       /-
                                                         🎉 no goals
                                                       -/


                                                          /-
                                                            a b : Prop
                                                            ⊢ Eq (Xor' a b) (Xor' b a)
                                                          -/
theorem xor_comm (a b : Prop) : Xor' a b = Xor' b a := by simp [Xor', and_comm, or_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


instance : Std.Commutative Xor' := ⟨xor_comm⟩


                                                             /-
                                                               a : Prop
                                                               ⊢ Eq (Xor' a a) False
                                                             -/
@[simp] theorem xor_self (a : Prop) : Xor' a a = False := by simp [Xor']
                                                             /-
                                                               🎉 no goals
                                                             -/


                                                           /-
                                                             a b : Prop
                                                             ⊢ Iff (Xor' (Not a) b) (Iff a b)
                                                           -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
@[simp] theorem xor_not_left : Xor' (¬a) b ↔ (a ↔ b) := by by_cases a <;> simp [*]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


                                                            /-
                                                              a b : Prop
                                                              ⊢ Iff (Xor' a (Not b)) (Iff a b)
                                                            -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
@[simp] theorem xor_not_right : Xor' a (¬b) ↔ (a ↔ b) := by by_cases a <;> simp [*]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


                                                      /-
                                                        a b : Prop
                                                        ⊢ Iff (Xor' (Not a) (Not b)) (Xor' a b)
                                                      -/
theorem xor_not_not : Xor' (¬a) (¬b) ↔ Xor' a b := by simp [Xor', or_comm, and_comm]
                                                      /-
                                                        🎉 no goals
                                                      -/


protected theorem Xor'.or (h : Xor' a b) : a ∨ b := h.imp And.left And.left


alias Iff.and := and_congr

alias ⟨And.rotate, _⟩ := and_rotate


                                                                                      /-
                                                                                        α : Sort u_1
                                                                                        a b : α
                                                                                        p : Prop
                                                                                        ⊢ Iff (And p (Eq a b)) (And p (Eq b a))
                                                                                      -/
theorem and_symm_right {α : Sort*} (a b : α) (p : Prop) : p ∧ a = b ↔ p ∧ b = a := by simp [eq_comm]
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/

                                                                                     /-
                                                                                       α : Sort u_1
                                                                                       a b : α
                                                                                       p : Prop
                                                                                       ⊢ Iff (And (Eq a b) p) (And (Eq b a) p)
                                                                                     -/
theorem and_symm_left {α : Sort*} (a b : α) (p : Prop) : a = b ∧ p ↔ b = a ∧ p := by simp [eq_comm]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


alias Iff.or := or_congr

alias ⟨Or.rotate, _⟩ := or_rotate


theorem Or.elim3 {c d : Prop} (h : a ∨ b ∨ c) (ha : a → d) (hb : b → d) (hc : c → d) : d :=
  Or.elim h ha fun h₂ ↦ Or.elim h₂ hb hc


theorem Or.imp3 {d e c f : Prop} (had : a → d) (hbe : b → e) (hcf : c → f) :
    a ∨ b ∨ c → d ∨ e ∨ f :=
  Or.imp had <| Or.imp hbe hcf


theorem not_or_of_imp : (a → b) → ¬a ∨ b := Decidable.not_or_of_imp

-- See Note [decidable namespace]

protected theorem Decidable.or_not_of_imp [Decidable a] (h : a → b) : b ∨ ¬a :=
  dite _ (Or.inl ∘ h) Or.inr


theorem or_not_of_imp : (a → b) → b ∨ ¬a := Decidable.or_not_of_imp


theorem imp_iff_not_or : a → b ↔ ¬a ∨ b := Decidable.imp_iff_not_or


theorem imp_iff_or_not {b a : Prop} : b → a ↔ a ∨ ¬b := Decidable.imp_iff_or_not


theorem not_imp_not : ¬a → ¬b ↔ b → a := Decidable.not_imp_not


                                                                        /-
                                                                          p q : Prop
                                                                          ⊢ Iff (And (p → q) (Not p → q)) q
                                                                        -/
theorem imp_and_neg_imp_iff (p q : Prop) : (p → q) ∧ (¬p → q) ↔ q := by simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


/-- Provide the reverse of modus tollens (`mt`) as dot notation for implications. -/
protected theorem Function.mtr : (¬a → ¬b) → b → a := not_imp_not.mp


theorem or_congr_left' {c a b : Prop} (h : ¬c → (a ↔ b)) : a ∨ c ↔ b ∨ c :=
  Decidable.or_congr_left' h


theorem or_congr_right' {c : Prop} (h : ¬a → (b ↔ c)) : a ∨ b ↔ a ∨ c := Decidable.or_congr_right' h


alias Iff.iff := iff_congr

-- @[simp] -- FIXME simp ignores proof rewrites

theorem iff_mpr_iff_true_intro {P : Prop} (h : P) : Iff.mpr (iff_true_intro h) True.intro = h := rfl


theorem imp_or {a b c : Prop} : a → b ∨ c ↔ (a → b) ∨ (a → c) := Decidable.imp_or


theorem imp_or' {a : Sort*} {b c : Prop} : a → b ∨ c ↔ (a → b) ∨ (a → c) := Decidable.imp_or'


theorem not_imp : ¬(a → b) ↔ a ∧ ¬b := Decidable.not_imp_iff_and_not


theorem peirce (a b : Prop) : ((a → b) → a) → a := Decidable.peirce _ _


theorem not_iff_not : (¬a ↔ ¬b) ↔ (a ↔ b) := Decidable.not_iff_not


theorem not_iff_comm : (¬a ↔ b) ↔ (¬b ↔ a) := Decidable.not_iff_comm


theorem not_iff : ¬(a ↔ b) ↔ (¬a ↔ b) := Decidable.not_iff


theorem iff_not_comm : (a ↔ ¬b) ↔ (b ↔ ¬a) := Decidable.iff_not_comm


theorem iff_iff_and_or_not_and_not : (a ↔ b) ↔ a ∧ b ∨ ¬a ∧ ¬b :=
  Decidable.iff_iff_and_or_not_and_not


theorem iff_iff_not_or_and_or_not : (a ↔ b) ↔ (¬a ∨ b) ∧ (a ∨ ¬b) :=
  Decidable.iff_iff_not_or_and_or_not


theorem not_and_not_right : ¬(a ∧ ¬b) ↔ a → b := Decidable.not_and_not_right


/-- One of **de Morgan's laws**: the negation of a conjunction is logically equivalent to the
disjunction of the negations. -/
theorem not_and_or : ¬(a ∧ b) ↔ ¬a ∨ ¬b := Decidable.not_and_iff_or_not_not


theorem or_iff_not_and_not : a ∨ b ↔ ¬(¬a ∧ ¬b) := Decidable.or_iff_not_and_not


theorem and_iff_not_or_not : a ∧ b ↔ ¬(¬a ∨ ¬b) := Decidable.and_iff_not_or_not


@[simp] theorem not_xor (P Q : Prop) : ¬Xor' P Q ↔ (P ↔ Q) := by
  /-
    P Q : Prop
    ⊢ Iff (Not (Xor' P Q)) (Iff P Q)
  -/
  simp only [not_and, Xor', not_or, not_not, ← iff_iff_implies_and_implies]
  /-
    🎉 no goals
  -/


theorem xor_iff_not_iff (P Q : Prop) : Xor' P Q ↔ ¬ (P ↔ Q) := (not_xor P Q).not_right


                                                    /-
                                                      a b : Prop
                                                      ⊢ Iff (Xor' a b) (Iff a (Not b))
                                                    -/
theorem xor_iff_iff_not : Xor' a b ↔ (a ↔ ¬b) := by simp only [← @xor_not_right a, not_not]
                                                    /-
                                                      🎉 no goals
                                                    -/


                                                     /-
                                                       a b : Prop
                                                       ⊢ Iff (Xor' a b) (Iff (Not a) b)
                                                     -/
theorem xor_iff_not_iff' : Xor' a b ↔ (¬a ↔ b) := by simp only [← @xor_not_left _ b, not_not]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem xor_iff_or_and_not_and (a b : Prop) : Xor' a b ↔ (a ∨ b) ∧ (¬ (a ∧ b)) := by
  rw [Xor', or_and_right, not_and_or, and_or_left, and_not_self_iff, false_or,
    and_or_left, and_not_self_iff, or_false]


alias Membership.mem.ne_of_not_mem := ne_of_mem_of_not_mem

alias Membership.mem.ne_of_not_mem' := ne_of_mem_of_not_mem'


theorem forall_cond_comm {α} {s : α → Prop} {p : α → α → Prop} :
    (∀ a, s a → ∀ b, s b → p a b) ↔ ∀ a b, s a → s b → p a b :=
  ⟨fun h a b ha hb ↦ h a ha b hb, fun h a ha b hb ↦ h a b ha hb⟩


theorem forall_mem_comm {α β} [Membership α β] {s : β} {p : α → α → Prop} :
    (∀ a (_ : a ∈ s) b (_ : b ∈ s), p a b) ↔ ∀ a b, a ∈ s → b ∈ s → p a b :=
  forall_cond_comm


@[deprecated (since := "2024-03-23")] alias ball_cond_comm := forall_cond_comm

@[deprecated (since := "2024-03-23")] alias ball_mem_comm := forall_mem_comm


lemma ne_of_eq_of_ne {α : Sort*} {a b c : α} (h₁ : a = b) (h₂ : b ≠ c) : a ≠ c := h₁.symm ▸ h₂

lemma ne_of_ne_of_eq {α : Sort*} {a b c : α} (h₁ : a ≠ b) (h₂ : b = c) : a ≠ c := h₂ ▸ h₁


alias Eq.trans_ne := ne_of_eq_of_ne

alias Ne.trans_eq := ne_of_ne_of_eq


theorem eq_equivalence {α : Sort*} : Equivalence (@Eq α) :=
  ⟨Eq.refl, @Eq.symm _, @Eq.trans _⟩

-- These were migrated to Batteries but the `@[simp]` attributes were (mysteriously?) removed.

theorem congr_refl_left {α β : Sort*} (f : α → β) {a b : α} (h : a = b) :
    congr (Eq.refl f) h = congr_arg f h := rfl

-- @[simp] -- FIXME simp ignores proof rewrites

theorem congr_refl_right {α β : Sort*} {f g : α → β} (h : f = g) (a : α) :
    congr h (Eq.refl a) = congr_fun h a := rfl

-- @[simp] -- FIXME simp ignores proof rewrites

theorem congr_arg_refl {α β : Sort*} (f : α → β) (a : α) :
    congr_arg f (Eq.refl a) = Eq.refl (f a) :=
  rfl

-- @[simp] -- FIXME simp ignores proof rewrites

theorem congr_fun_rfl {α β : Sort*} (f : α → β) (a : α) : congr_fun (Eq.refl f) a = Eq.refl (f a) :=
  rfl

-- @[simp] -- FIXME simp ignores proof rewrites

theorem congr_fun_congr_arg {α β γ : Sort*} (f : α → β → γ) {a a' : α} (p : a = a') (b : β) :
    congr_fun (congr_arg f p) b = congr_arg (fun a ↦ f a b) p := rfl


theorem Eq.rec_eq_cast {α : Sort _} {P : α → Sort _} {x y : α} (h : x = y) (z : P x) :
                                         /-
                                           α : Sort u_1
                                           P : α → Sort u_2
                                           x y : α
                                           h : Eq x y
                                           z : P x
                                           ⊢ Eq (Eq.rec z h) (cast ⋯ z)
                                         -/
    h ▸ z = cast (congr_arg P h) z := by induction h; rfl
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem eqRec_heq' {α : Sort*} {a' : α} {motive : (a : α) → a' = a → Sort*}
    (p : motive a' (rfl : a' = a')) {a : α} (t : a' = a) :
    HEq (@Eq.rec α a' motive p a t) p := by
  /-
    α : Sort u_1
    a' : α
    motive : (a : α) → Eq a' a → Sort u_2
    p : motive a' ⋯
    a : α
    t : Eq a' a
    ⊢ HEq (Eq.rec p t) p
  -/
  subst t; rfl
           /-
             🎉 no goals
           -/


theorem rec_heq_of_heq {α β : Sort _} {a b : α} {C : α → Sort*} {x : C a} {y : β}
                                                    /-
                                                      α : Sort u_2
                                                      β : Sort u_1
                                                      a b : α
                                                      C : α → Sort u_1
                                                      x : C a
                                                      y : β
                                                      e : Eq a b
                                                      h : HEq x y
                                                      ⊢ HEq (Eq.rec x e) y
                                                    -/
    (e : a = b) (h : HEq x y) : HEq (e ▸ x) y := by subst e; exact h
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem rec_heq_iff_heq {α β : Sort _} {a b : α} {C : α → Sort*} {x : C a} {y : β} {e : a = b} :
                                  /-
                                    α : Sort u_2
                                    β : Sort u_1
                                    a b : α
                                    C : α → Sort u_1
                                    x : C a
                                    y : β
                                    e : Eq a b
                                    ⊢ Iff (HEq (Eq.rec x e) y) (HEq x y)
                                  -/
    HEq (e ▸ x) y ↔ HEq x y := by subst e; rfl
                                           /-
                                             🎉 no goals
                                           -/


theorem heq_rec_iff_heq {α β : Sort _} {a b : α} {C : α → Sort*} {x : β} {y : C a} {e : a = b} :
                                  /-
                                    α : Sort u_2
                                    β : Sort u_1
                                    a b : α
                                    C : α → Sort u_1
                                    x : β
                                    y : C a
                                    e : Eq a b
                                    ⊢ Iff (HEq x (Eq.rec y e)) (HEq x y)
                                  -/
    HEq x (e ▸ y) ↔ HEq x y := by subst e; rfl
                                           /-
                                             🎉 no goals
                                           -/


                                                                /-
                                                                  α β : Sort u
                                                                  a : α
                                                                  b : β
                                                                  e : Eq β α
                                                                  ⊢ Eq a (cast e b) → HEq a b
                                                                -/
lemma heq_of_eq_cast (e : β = α) : a = cast e b → HEq a b := by rintro rfl; simp
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


                                                                                /-
                                                                                  α β : Sort u
                                                                                  e : Eq β α
                                                                                  a : α
                                                                                  b : β
                                                                                  h : HEq a b
                                                                                  ⊢ Eq a (cast e b)
                                                                                -/
lemma eq_cast_iff_heq : a = cast e b ↔ HEq a b := ⟨heq_of_eq_cast _, fun h ↦ by cases h; rfl⟩
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


theorem forall₂_imp {p q : ∀ a, β a → Prop} (h : ∀ a b, p a b → q a b) :
    (∀ a b, p a b) → ∀ a b, q a b :=
  forall_imp fun i ↦ forall_imp <| h i


theorem forall₃_imp {p q : ∀ a b, γ a b → Prop} (h : ∀ a b c, p a b c → q a b c) :
    (∀ a b c, p a b c) → ∀ a b c, q a b c :=
  forall_imp fun a ↦ forall₂_imp <| h a


theorem Exists₂.imp {p q : ∀ a, β a → Prop} (h : ∀ a b, p a b → q a b) :
    (∃ a b, p a b) → ∃ a b, q a b :=
  Exists.imp fun a ↦ Exists.imp <| h a


theorem Exists₃.imp {p q : ∀ a b, γ a b → Prop} (h : ∀ a b c, p a b c → q a b c) :
    (∃ a b c, p a b c) → ∃ a b c, q a b c :=
  Exists.imp fun a ↦ Exists₂.imp <| h a


theorem forall_swap {p : α → β → Prop} : (∀ x y, p x y) ↔ ∀ y x, p x y :=
  ⟨fun f x y ↦ f y x, fun f x y ↦ f y x⟩


theorem forall₂_swap
    {ι₁ ι₂ : Sort*} {κ₁ : ι₁ → Sort*} {κ₂ : ι₂ → Sort*} {p : ∀ i₁, κ₁ i₁ → ∀ i₂, κ₂ i₂ → Prop} :
    (∀ i₁ j₁ i₂ j₂, p i₁ j₁ i₂ j₂) ↔ ∀ i₂ j₂ i₁ j₁, p i₁ j₁ i₂ j₂ := ⟨swap₂, swap₂⟩


/-- We intentionally restrict the type of `α` in this lemma so that this is a safer to use in simp
than `forall_swap`. -/
theorem imp_forall_iff {α : Type*} {p : Prop} {q : α → Prop} : (p → ∀ x, q x) ↔ ∀ x, p → q x :=
  forall_swap


@[simp] lemma imp_forall_iff_forall (A : Prop) (B : A → Prop) :
                                          /-
                                            A : Prop
                                            B : A → Prop
                                            ⊢ Iff (A → ∀ (h : A), B h) (∀ (h : A), B h)
                                          -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  (A → ∀ h : A, B h) ↔ ∀ h : A, B h := by by_cases h : A <;> simp [h]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem exists_swap {p : α → β → Prop} : (∃ x y, p x y) ↔ ∃ y x, p x y :=
  ⟨fun ⟨x, y, h⟩ ↦ ⟨y, x, h⟩, fun ⟨y, x, h⟩ ↦ ⟨x, y, h⟩⟩


theorem exists_and_exists_comm {P : α → Prop} {Q : β → Prop} :
    (∃ a, P a) ∧ (∃ b, Q b) ↔ ∃ a b, P a ∧ Q b :=
  ⟨fun ⟨⟨a, ha⟩, ⟨b, hb⟩⟩ ↦ ⟨a, b, ⟨ha, hb⟩⟩, fun ⟨a, b, ⟨ha, hb⟩⟩ ↦ ⟨⟨a, ha⟩, ⟨b, hb⟩⟩⟩


theorem not_forall_not : (¬∀ x, ¬p x) ↔ ∃ x, p x := Decidable.not_forall_not


lemma forall_or_exists_not (P : α → Prop) : (∀ a, P a) ∨ ∃ a, ¬ P a := by
  /-
    α : Sort u_1
    P : α → Prop
    ⊢ Or (∀ (a : α), P a) (Exists fun a => Not (P a))
  -/
  rw [← not_forall]; exact em _
                     /-
                       🎉 no goals
                     -/


lemma exists_or_forall_not (P : α → Prop) : (∃ a, P a) ∨ ∀ a, ¬ P a := by
  /-
    α : Sort u_1
    P : α → Prop
    ⊢ Or (Exists fun a => P a) (∀ (a : α), Not (P a))
  -/
  rw [← not_exists]; exact em _
                     /-
                       🎉 no goals
                     -/


theorem forall_imp_iff_exists_imp {α : Sort*} {p : α → Prop} {b : Prop} [ha : Nonempty α] :
    (∀ x, p x) → b ↔ ∃ x, p x → b := by
  /-
    α : Sort u_3
    p : α → Prop
    b : Prop
    ha : Nonempty α
    ⊢ Iff ((∀ (x : α), p x) → b) (Exists fun x => p x → b)
  -/
  let ⟨a⟩ := ha
  /-
    α : Sort u_3
    p : α → Prop
    b : Prop
    ha : Nonempty α
    a : α
    ⊢ Iff ((∀ (x : α), p x) → b) (Exists fun x => p x → b)
  -/
  refine ⟨fun h ↦ not_forall_not.1 fun h' ↦ ?_, fun ⟨x, hx⟩ h ↦ hx (h x)⟩
  /-
    α : Sort u_3
    p : α → Prop
    b : Prop
    ha : Nonempty α
    a : α
    h : (∀ (x : α), p x) → b
    h' : ∀ (x : α), Not (p x → b)
    ⊢ False
  -/
  exact if hb : b then h' a fun _ ↦ hb else hb <| h fun x ↦ (_root_.not_imp.1 (h' x)).1
  /-
    🎉 no goals
  -/


@[mfld_simps]
theorem forall_true_iff : (α → True) ↔ True := imp_true_iff _

-- Unfortunately this causes simp to loop sometimes, so we
-- add the 2 and 3 cases as simp lemmas instead

theorem forall_true_iff' (h : ∀ a, p a ↔ True) : (∀ a, p a) ↔ True :=
  iff_true_intro fun _ ↦ of_iff_true (h _)

-- This is not marked `@[simp]` because `implies_true : (α → True) = True` works

                                                                          /-
                                                                            α : Sort u_1
                                                                            β : α → Sort u_3
                                                                            ⊢ Iff (∀ (a : α), β a → True) True
                                                                          -/
theorem forall₂_true_iff {β : α → Sort*} : (∀ a, β a → True) ↔ True := by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/

-- This is not marked `@[simp]` because `implies_true : (α → True) = True` works

theorem forall₃_true_iff {β : α → Sort*} {γ : ∀ a, β a → Sort*} :
                                                 /-
                                                   α : Sort u_1
                                                   β : α → Sort u_3
                                                   γ : (a : α) → β a → Sort u_4
                                                   ⊢ Iff (∀ (a : α) (b : β a), γ a b → True) True
                                                 -/
    (∀ (a) (b : β a), γ a b → True) ↔ True := by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem Decidable.and_forall_ne [DecidableEq α] (a : α) {p : α → Prop} :
    (p a ∧ ∀ b, b ≠ a → p b) ↔ ∀ b, p b := by
  /-
    α : Sort u_1
    inst✝ : DecidableEq α
    a : α
    p : α → Prop
    ⊢ Iff (And (p a) (∀ (b : α), Ne b a → p b)) (∀ (b : α), p b)
  -/
  simp only [← @forall_eq _ p a, ← forall_and, ← or_imp, Decidable.em, forall_const]
  /-
    🎉 no goals
  -/


theorem and_forall_ne (a : α) : (p a ∧ ∀ b, b ≠ a → p b) ↔ ∀ b, p b :=
  Decidable.and_forall_ne a


theorem Ne.ne_or_ne {x y : α} (z : α) (h : x ≠ y) : x ≠ z ∨ y ≠ z :=
  not_and_or.1 <| mt (and_imp.2 (· ▸ ·)) h.symm


@[simp]
theorem exists_apply_eq_apply' (f : α → β) (a' : α) : ∃ a, f a' = f a := ⟨a', rfl⟩


@[simp]
lemma exists_apply_eq_apply2 {α β γ} {f : α → β → γ} {a : α} {b : β} : ∃ x y, f x y = f a b :=
  ⟨a, b, rfl⟩


@[simp]
lemma exists_apply_eq_apply2' {α β γ} {f : α → β → γ} {a : α} {b : β} : ∃ x y, f a b = f x y :=
  ⟨a, b, rfl⟩


@[simp]
lemma exists_apply_eq_apply3 {α β γ δ} {f : α → β → γ → δ} {a : α} {b : β} {c : γ} :
    ∃ x y z, f x y z = f a b c :=
  ⟨a, b, c, rfl⟩


@[simp]
lemma exists_apply_eq_apply3' {α β γ δ} {f : α → β → γ → δ} {a : α} {b : β} {c : γ} :
    ∃ x y z, f a b c = f x y z :=
  ⟨a, b, c, rfl⟩

-- Porting note: an alternative workaround theorem:

theorem exists_apply_eq (a : α) (b : β) : ∃ f : α → β, f a = b := ⟨fun _ ↦ b, rfl⟩


@[simp] theorem exists_exists_and_eq_and {f : α → β} {p : α → Prop} {q : β → Prop} :
    (∃ b, (∃ a, p a ∧ f a = b) ∧ q b) ↔ ∃ a, p a ∧ q (f a) :=
  ⟨fun ⟨_, ⟨a, ha, hab⟩, hb⟩ ↦ ⟨a, ha, hab.symm ▸ hb⟩, fun ⟨a, hp, hq⟩ ↦ ⟨f a, ⟨a, hp, rfl⟩, hq⟩⟩


@[simp] theorem exists_exists_eq_and {f : α → β} {p : β → Prop} :
    (∃ b, (∃ a, f a = b) ∧ p b) ↔ ∃ a, p (f a) :=
  ⟨fun ⟨_, ⟨a, ha⟩, hb⟩ ↦ ⟨a, ha.symm ▸ hb⟩, fun ⟨a, ha⟩ ↦ ⟨f a, ⟨a, rfl⟩, ha⟩⟩


@[simp] theorem exists_exists_and_exists_and_eq_and {α β γ : Type*}
    {f : α → β → γ} {p : α → Prop} {q : β → Prop} {r : γ → Prop} :
    (∃ c, (∃ a, p a ∧ ∃ b, q b ∧ f a b = c) ∧ r c) ↔ ∃ a, p a ∧ ∃ b, q b ∧ r (f a b) :=
  ⟨fun ⟨_, ⟨a, ha, b, hb, hab⟩, hc⟩ ↦ ⟨a, ha, b, hb, hab.symm ▸ hc⟩,
    fun ⟨a, ha, b, hb, hab⟩ ↦ ⟨f a b, ⟨a, ha, b, hb, rfl⟩, hab⟩⟩


@[simp] theorem exists_exists_exists_and_eq {α β γ : Type*}
    {f : α → β → γ} {p : γ → Prop} :
    (∃ c, (∃ a, ∃ b, f a b = c) ∧ p c) ↔ ∃ a, ∃ b, p (f a b) :=
  ⟨fun ⟨_, ⟨a, b, hab⟩, hc⟩ ↦ ⟨a, b, hab.symm ▸ hc⟩,
    fun ⟨a, b, hab⟩ ↦ ⟨f a b, ⟨a, b, rfl⟩, hab⟩⟩


theorem forall_apply_eq_imp_iff' {f : α → β} {p : β → Prop} :
                                                /-
                                                  α : Sort u_1
                                                  β : Sort u_2
                                                  f : α → β
                                                  p : β → Prop
                                                  ⊢ Iff (∀ (a : α) (b : β), Eq (f a) b → p b) (∀ (a : α), p (f a))
                                                -/
    (∀ a b, f a = b → p b) ↔ ∀ a, p (f a) := by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem forall_eq_apply_imp_iff' {f : α → β} {p : β → Prop} :
                                                /-
                                                  α : Sort u_1
                                                  β : Sort u_2
                                                  f : α → β
                                                  p : β → Prop
                                                  ⊢ Iff (∀ (a : α) (b : β), Eq b (f a) → p b) (∀ (a : α), p (f a))
                                                -/
    (∀ a b, b = f a → p b) ↔ ∀ a, p (f a) := by simp
                                                /-
                                                  🎉 no goals
                                                -/


theorem exists₂_comm
    {ι₁ ι₂ : Sort*} {κ₁ : ι₁ → Sort*} {κ₂ : ι₂ → Sort*} {p : ∀ i₁, κ₁ i₁ → ∀ i₂, κ₂ i₂ → Prop} :
    (∃ i₁ j₁ i₂ j₂, p i₁ j₁ i₂ j₂) ↔ ∃ i₂ j₂ i₁ j₁, p i₁ j₁ i₂ j₂ := by
  /-
    ι₁ : Sort u_3
    ι₂ : Sort u_4
    κ₁ : ι₁ → Sort u_5
    κ₂ : ι₂ → Sort u_6
    p : (i₁ : ι₁) → κ₁ i₁ → (i₂ : ι₂) → κ₂ i₂ → Prop
    ⊢ Iff (Exists fun i₁ => Exists fun j₁ => Exists fun i₂ => Exists fun j₂ => p i …
  -/
  simp only [@exists_comm (κ₁ _), @exists_comm ι₁]
  /-
    🎉 no goals
  -/


theorem And.exists {p q : Prop} {f : p ∧ q → Prop} : (∃ h, f h) ↔ ∃ hp hq, f ⟨hp, hq⟩ :=
  ⟨fun ⟨h, H⟩ ↦ ⟨h.1, h.2, H⟩, fun ⟨hp, hq, H⟩ ↦ ⟨⟨hp, hq⟩, H⟩⟩


theorem forall_or_of_or_forall {α : Sort*} {p : α → Prop} {b : Prop} (h : b ∨ ∀ x, p x) (x : α) :
    b ∨ p x :=
  h.imp_right fun h₂ ↦ h₂ x

-- See Note [decidable namespace]

protected theorem Decidable.forall_or_left {q : Prop} {p : α → Prop} [Decidable q] :
    (∀ x, q ∨ p x) ↔ q ∨ ∀ x, p x :=
  ⟨fun h ↦ if hq : q then Or.inl hq else
    Or.inr fun x ↦ (h x).resolve_left hq, forall_or_of_or_forall⟩


theorem forall_or_left {q} {p : α → Prop} : (∀ x, q ∨ p x) ↔ q ∨ ∀ x, p x :=
  Decidable.forall_or_left

-- See Note [decidable namespace]

protected theorem Decidable.forall_or_right {q} {p : α → Prop} [Decidable q] :
                                          /-
                                            α : Sort u_1
                                            q : Prop
                                            p : α → Prop
                                            inst✝ : Decidable q
                                            ⊢ Iff (∀ (x : α), Or (p x) q) (Or (∀ (x : α), p x) q)
                                          -/
    (∀ x, p x ∨ q) ↔ (∀ x, p x) ∨ q := by simp [or_comm, Decidable.forall_or_left]
                                          /-
                                            🎉 no goals
                                          -/


theorem forall_or_right {q} {p : α → Prop} : (∀ x, p x ∨ q) ↔ (∀ x, p x) ∨ q :=
  Decidable.forall_or_right


theorem Exists.fst {b : Prop} {p : b → Prop} : Exists p → b
  | ⟨h, _⟩ => h


theorem Exists.snd {b : Prop} {p : b → Prop} : ∀ h : Exists p, p h.fst
  | ⟨_, h⟩ => h


theorem Prop.exists_iff {p : Prop → Prop} : (∃ h, p h) ↔ p False ∨ p True :=
                                                    /-
                                                      p : Prop → Prop
                                                      x✝ : Exists fun h => p h
                                                      h₁ : Prop
                                                      h₂ : p h₁
                                                      H : h₁
                                                      ⊢ p True
                                                    -/
  ⟨fun ⟨h₁, h₂⟩ ↦ by_cases (fun H : h₁ ↦ .inr <| by simpa only [H] using h₂)
                                                    /-
                                                      🎉 no goals
                                                    -/
                        /-
                          p : Prop → Prop
                          x✝ : Exists fun h => p h
                          h₁ : Prop
                          h₂ : p h₁
                          H : Not h₁
                          ⊢ p False
                        -/
    (fun H ↦ .inl <| by simpa only [H] using h₂), fun h ↦ h.elim (.intro _) (.intro _)⟩
                        /-
                          🎉 no goals
                        -/


theorem Prop.forall_iff {p : Prop → Prop} : (∀ h, p h) ↔ p False ∧ p True :=
                                           /-
                                             p : Prop → Prop
                                             x✝ : And (p False) (p True)
                                             h : Prop
                                             h₁ : p False
                                             h₂ : p True
                                             ⊢ p h
                                           -/
                                                              /-
                                                                🎉 no goals
                                                              -/
  ⟨fun H ↦ ⟨H _, H _⟩, fun ⟨h₁, h₂⟩ h ↦ by by_cases H : h <;> simpa only [H]⟩
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem exists_iff_of_forall {p : Prop} {q : p → Prop} (h : ∀ h, q h) : (∃ h, q h) ↔ p :=
  ⟨Exists.fst, fun H ↦ ⟨H, h H⟩⟩


theorem exists_prop_of_false {p : Prop} {q : p → Prop} : ¬p → ¬∃ h' : p, q h' :=
  mt Exists.fst

/- See `IsEmpty.exists_iff` for the `False` version of `exists_true_left`. -/

-- Porting note: `@[congr]` commented out for now.
-- @[congr]

theorem forall_prop_congr {p p' : Prop} {q q' : p → Prop} (hq : ∀ h, q h ↔ q' h) (hp : p ↔ p') :
    (∀ h, q h) ↔ ∀ h : p', q' (hp.2 h) :=
  ⟨fun h1 h2 ↦ (hq _).1 (h1 (hp.2 h2)), fun h1 h2 ↦ (hq _).2 (h1 (hp.1 h2))⟩

-- Porting note: `@[congr]` commented out for now.
-- @[congr]

theorem forall_prop_congr' {p p' : Prop} {q q' : p → Prop} (hq : ∀ h, q h ↔ q' h) (hp : p ↔ p') :
    (∀ h, q h) = ∀ h : p', q' (hp.2 h) :=
  propext (forall_prop_congr hq hp)


lemma imp_congr_eq {a b c d : Prop} (h₁ : a = c) (h₂ : b = d) : (a → b) = (c → d) :=
  propext (imp_congr h₁.to_iff h₂.to_iff)


lemma imp_congr_ctx_eq {a b c d : Prop} (h₁ : a = c) (h₂ : c → b = d) : (a → b) = (c → d) :=
  propext (imp_congr_ctx h₁.to_iff fun hc ↦ (h₂ hc).to_iff)


lemma eq_true_intro {a : Prop} (h : a) : a = True := propext (iff_true_intro h)


lemma eq_false_intro {a : Prop} (h : ¬a) : a = False := propext (iff_false_intro h)

-- FIXME: `alias` creates `def Iff.eq := propext` instead of `lemma Iff.eq := propext`

@[nolint defLemma] alias Iff.eq := propext


lemma iff_eq_eq {a b : Prop} : (a ↔ b) = (a = b) := propext ⟨propext, Eq.to_iff⟩

-- They were not used in Lean 3 and there are already lemmas with those names in Lean 4


/-- See `IsEmpty.forall_iff` for the `False` version. -/
@[simp] theorem forall_true_left (p : True → Prop) : (∀ x, p x) ↔ p True.intro :=
  forall_prop_of_true _


/-- Any prop `p` is decidable classically. A shorthand for `Classical.propDecidable`. -/
                                                     /-
                                                       p : Prop
                                                       ⊢ Decidable p
                                                     -/
noncomputable def dec (p : Prop) : Decidable p := by infer_instance
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- Any predicate `p` is decidable classically. -/
                                                                 /-
                                                                   α : Sort u_1
                                                                   p : α → Prop
                                                                   ⊢ DecidablePred p
                                                                 -/
noncomputable def decPred (p : α → Prop) : DecidablePred p := by infer_instance
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


/-- Any relation `p` is decidable classically. -/
                                                                   /-
                                                                     α : Sort u_1
                                                                     p : α → α → Prop
                                                                     ⊢ DecidableRel p
                                                                   -/
noncomputable def decRel (p : α → α → Prop) : DecidableRel p := by infer_instance
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Any type `α` has decidable equality classically. -/
                                                          /-
                                                            α✝ : Sort u_1
                                                            α : Sort u_2
                                                            ⊢ DecidableEq α
                                                          -/
noncomputable def decEq (α : Sort*) : DecidableEq α := by infer_instance
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Construct a function from a default value `H0`, and a function to use if there exists a value
satisfying the predicate. -/
noncomputable def existsCases {α C : Sort*} {p : α → Prop} (H0 : C) (H : ∀ a, p a → C) : C :=
  if h : ∃ a, p a then H (Classical.choose h) (Classical.choose_spec h) else H0


theorem some_spec₂ {α : Sort*} {p : α → Prop} {h : ∃ a, p a} (q : α → Prop)
    (hpq : ∀ a, p a → q a) : q (choose h) := hpq _ <| choose_spec _


/-- A version of `Classical.indefiniteDescription` which is definitionally equal to a pair.

In Lean 4, this definition is defeq to `Classical.indefiniteDescription`,
so it is deprecated. -/
@[deprecated Classical.indefiniteDescription (since := "2024-07-04")]
noncomputable def subtype_of_exists {α : Type*} {P : α → Prop} (h : ∃ x, P x) : { x // P x } :=
  ⟨Classical.choose h, Classical.choose_spec h⟩


/-- A version of `byContradiction` that uses types instead of propositions. -/
protected noncomputable def byContradiction' {α : Sort*} (H : ¬(α → False)) : α :=
  Classical.choice <| (peirce _ False) fun h ↦ (H fun a ↦ h ⟨a⟩).elim


/-- `Classical.byContradiction'` is equivalent to lean's axiom `Classical.choice`. -/
def choice_of_byContradiction' {α : Sort*} (contra : ¬(α → False) → α) : Nonempty α → α :=
  fun H ↦ contra H.elim


@[simp] lemma choose_eq (a : α) : @Exists.choose _ (· = a) ⟨a, rfl⟩ = a := @choose_spec _ (· = a) _


@[simp]
lemma choose_eq' (a : α) : @Exists.choose _ (a = ·) ⟨a, rfl⟩ = a :=
  (@choose_spec _ (a = ·) _).symm


alias axiom_of_choice := axiomOfChoice -- TODO: remove? rename in core?

alias by_cases := byCases -- TODO: remove? rename in core?

alias by_contradiction := byContradiction -- TODO: remove? rename in core?

-- The remaining theorems in this section were ported from Lean 3,
-- but are currently unused in Mathlib, so have been deprecated.
-- If any are being used downstream, please remove the deprecation.


alias prop_complete := propComplete -- TODO: remove? rename in core?


@[elab_as_elim, deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem cases_true_false (p : Prop → Prop)
    (h1 : p True) (h2 : p False) (a : Prop) : p a :=
  Or.elim (prop_complete a) (fun ht : a = True ↦ ht.symm ▸ h1) fun hf : a = False ↦ hf.symm ▸ h2


@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem eq_false_or_eq_true (a : Prop) : a = False ∨ a = True := (prop_complete a).symm


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem cases_on (a : Prop) {p : Prop → Prop} (h1 : p True) (h2 : p False) : p a :=
  @cases_true_false p h1 h2 a


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided." (since := "2024-07-27")]
theorem cases {p : Prop → Prop} (h1 : p True) (h2 : p False) (a) : p a := cases_on a h1 h2


/-- This function has the same type as `Exists.recOn`, and can be used to case on an equality,
but `Exists.recOn` can only eliminate into Prop, while this version eliminates into any universe
using the axiom of choice. -/
noncomputable def Exists.classicalRecOn {α : Sort*} {p : α → Prop} (h : ∃ a, p a)
    {C : Sort*} (H : ∀ a, p a → C) : C :=
  H (Classical.choose h) (Classical.choose_spec h)


theorem bex_def : (∃ (x : _) (_ : p x), q x) ↔ ∃ x, p x ∧ q x :=
  ⟨fun ⟨x, px, qx⟩ ↦ ⟨x, px, qx⟩, fun ⟨x, px, qx⟩ ↦ ⟨x, px, qx⟩⟩


theorem BEx.elim {b : Prop} : (∃ x h, P x h) → (∀ a h, P a h → b) → b
  | ⟨a, h₁, h₂⟩, h' => h' a h₁ h₂


theorem BEx.intro (a : α) (h₁ : p a) (h₂ : P a h₁) : ∃ (x : _) (h : p x), P x h :=
  ⟨a, h₁, h₂⟩


@[deprecated exists_eq_left (since := "2024-04-06")]
theorem bex_eq_left {a : α} : (∃ (x : _) (_ : x = a), p x) ↔ p a := by
  /-
    α : Sort u_1
    p : α → Prop
    a : α
    ⊢ Iff (Exists fun x => Exists fun x_1 => p x) (p a)
  -/
  simp only [exists_prop, exists_eq_left]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-06")] alias ball_congr := forall₂_congr

@[deprecated (since := "2024-04-06")] alias bex_congr := exists₂_congr


theorem BAll.imp_right (H : ∀ x h, P x h → Q x h) (h₁ : ∀ x h, P x h) (x h) : Q x h :=
  H _ _ <| h₁ _ _


theorem BEx.imp_right (H : ∀ x h, P x h → Q x h) : (∃ x h, P x h) → ∃ x h, Q x h
  | ⟨_, _, h'⟩ => ⟨_, _, H _ _ h'⟩


theorem BAll.imp_left (H : ∀ x, p x → q x) (h₁ : ∀ x, q x → r x) (x) (h : p x) : r x :=
  h₁ _ <| H _ h


theorem BEx.imp_left (H : ∀ x, p x → q x) : (∃ (x : _) (_ : p x), r x) → ∃ (x : _) (_ : q x), r x
  | ⟨x, hp, hr⟩ => ⟨x, H _ hp, hr⟩


@[deprecated id (since := "2024-03-23")]
theorem ball_of_forall (h : ∀ x, p x) (x) : p x := h x


@[deprecated forall_imp (since := "2024-03-23")]
theorem forall_of_ball (H : ∀ x, p x) (h : ∀ x, p x → q x) (x) : q x := h x <| H x


theorem exists_mem_of_exists (H : ∀ x, p x) : (∃ x, q x) → ∃ (x : _) (_ : p x), q x
  | ⟨x, hq⟩ => ⟨x, H x, hq⟩


theorem exists_of_exists_mem : (∃ (x : _) (_ : p x), q x) → ∃ x, q x
  | ⟨x, _, hq⟩ => ⟨x, hq⟩


@[deprecated (since := "2024-03-23")] alias bex_of_exists := exists_mem_of_exists

@[deprecated (since := "2024-03-23")] alias exists_of_bex := exists_of_exists_mem

@[deprecated (since := "2024-03-23")] alias bex_imp := exists₂_imp


theorem not_exists_mem : (¬∃ x h, P x h) ↔ ∀ x h, ¬P x h := exists₂_imp


theorem not_forall₂_of_exists₂_not : (∃ x h, ¬P x h) → ¬∀ x h, P x h
  | ⟨x, h, hp⟩, al => hp <| al x h

-- See Note [decidable namespace]

protected theorem Decidable.not_forall₂ [Decidable (∃ x h, ¬P x h)] [∀ x h, Decidable (P x h)] :
    (¬∀ x h, P x h) ↔ ∃ x h, ¬P x h :=
  ⟨Not.decidable_imp_symm fun nx x h ↦ nx.decidable_imp_symm
    fun h' ↦ ⟨x, h, h'⟩, not_forall₂_of_exists₂_not⟩


theorem not_forall₂ : (¬∀ x h, P x h) ↔ ∃ x h, ¬P x h := Decidable.not_forall₂


theorem forall₂_and : (∀ x h, P x h ∧ Q x h) ↔ (∀ x h, P x h) ∧ ∀ x h, Q x h :=
  Iff.trans (forall_congr' fun _ ↦ forall_and) forall_and


theorem forall_and_left [Nonempty α] (q : Prop) (p : α → Prop) :
                                          /-
                                            α : Sort u_1
                                            inst✝ : Nonempty α
                                            q : Prop
                                            p : α → Prop
                                            ⊢ Iff (∀ (x : α), And q (p x)) (And q (∀ (x : α), p x))
                                          -/
    (∀ x, q ∧ p x) ↔ (q ∧ ∀ x, p x) := by rw [forall_and, forall_const]
                                          /-
                                            🎉 no goals
                                          -/


theorem forall_and_right [Nonempty α] (p : α → Prop) (q : Prop) :
                                          /-
                                            α : Sort u_1
                                            inst✝ : Nonempty α
                                            p : α → Prop
                                            q : Prop
                                            ⊢ Iff (∀ (x : α), And (p x) q) (And (∀ (x : α), p x) q)
                                          -/
    (∀ x, p x ∧ q) ↔ (∀ x, p x) ∧ q := by rw [forall_and, forall_const]
                                          /-
                                            🎉 no goals
                                          -/


theorem exists_mem_or : (∃ x h, P x h ∨ Q x h) ↔ (∃ x h, P x h) ∨ ∃ x h, Q x h :=
  Iff.trans (exists_congr fun _ ↦ exists_or) exists_or


theorem forall₂_or_left : (∀ x, p x ∨ q x → r x) ↔ (∀ x, p x → r x) ∧ ∀ x, q x → r x :=
  Iff.trans (forall_congr' fun _ ↦ or_imp) forall_and


theorem exists_mem_or_left :
    (∃ (x : _) (_ : p x ∨ q x), r x) ↔ (∃ (x : _) (_ : p x), r x) ∨ ∃ (x : _) (_ : q x), r x := by
  /-
    α : Sort u_1
    r p q : α → Prop
    ⊢ Iff (Exists fun x => Exists fun x_1 => r x) (Or (Exists fun x => Exists fun  …
  -/
  simp only [exists_prop]
  /-
    α : Sort u_1
    r p q : α → Prop
    ⊢ Iff (Exists fun x => And (Or (p x) (q x)) (r x)) (Or (Exists fun x => And (p …
  -/
  exact Iff.trans (exists_congr fun x ↦ or_and_right) exists_or
  /-
    🎉 no goals
  -/


theorem dite_eq_iff : dite P A B = c ↔ (∃ h, A h = c) ∨ ∃ h, B h = c := by
  /-
    α : Sort u_1
    P : Prop
    inst✝ : Decidable P
    c : α
    A : P → α
    B : Not P → α
    ⊢ Iff (Eq (dite P A B) c) (Or (Exists fun h => Eq (A h) c) (Exists fun h => Eq …
  -/
                 /-
                   🎉 no goals
                 -/
  by_cases P <;> simp [*, exists_prop_of_true, exists_prop_of_false]
                 /-
                   🎉 no goals
                 -/


theorem ite_eq_iff : ite P a b = c ↔ P ∧ a = c ∨ ¬P ∧ b = c :=
                          /-
                            α : Sort u_1
                            P : Prop
                            inst✝ : Decidable P
                            a b c : α
                            ⊢ Iff (Or (Exists fun h => Eq a c) (Exists fun h => Eq b c)) (Or (And P (Eq a  …
                          -/
  dite_eq_iff.trans <| by rw [exists_prop, exists_prop]
                          /-
                            🎉 no goals
                          -/


theorem eq_ite_iff : a = ite P b c ↔ P ∧ a = b ∨ ¬P ∧ a = c :=
  eq_comm.trans <| ite_eq_iff.trans <| (Iff.rfl.and eq_comm).or (Iff.rfl.and eq_comm)


theorem dite_eq_iff' : dite P A B = c ↔ (∀ h, A h = c) ∧ ∀ h, B h = c :=
  ⟨fun he ↦ ⟨fun h ↦ (dif_pos h).symm.trans he, fun h ↦ (dif_neg h).symm.trans he⟩, fun he ↦
    (em P).elim (fun h ↦ (dif_pos h).trans <| he.1 h) fun h ↦ (dif_neg h).trans <| he.2 h⟩


theorem ite_eq_iff' : ite P a b = c ↔ (P → a = c) ∧ (¬P → b = c) := dite_eq_iff'


theorem dite_ne_left_iff : dite P (fun _ ↦ a) B ≠ a ↔ ∃ h, a ≠ B h := by
  /-
    α : Sort u_1
    P : Prop
    inst✝ : Decidable P
    a : α
    B : Not P → α
    ⊢ Iff (Ne (dite P (fun x => a) B) a) (Exists fun h => Ne a (B h))
  -/
  rw [Ne, dite_eq_left_iff, not_forall]
  /-
    α : Sort u_1
    P : Prop
    inst✝ : Decidable P
    a : α
    B : Not P → α
    ⊢ Iff (Exists fun x => Not (Eq (B x) a)) (Exists fun h => Ne a (B h))
  -/
  exact exists_congr fun h ↦ by rw [ne_comm]
  /-
    🎉 no goals
  -/


theorem dite_ne_right_iff : (dite P A fun _ ↦ b) ≠ b ↔ ∃ h, A h ≠ b := by
  /-
    α : Sort u_1
    P : Prop
    inst✝ : Decidable P
    b : α
    A : P → α
    ⊢ Iff (Ne (dite P A fun x => b) b) (Exists fun h => Ne (A h) b)
  -/
  simp only [Ne, dite_eq_right_iff, not_forall]
  /-
    🎉 no goals
  -/


theorem ite_ne_left_iff : ite P a b ≠ a ↔ ¬P ∧ a ≠ b :=
                               /-
                                 α : Sort u_1
                                 P : Prop
                                 inst✝ : Decidable P
                                 a b : α
                                 ⊢ Iff (Exists fun h => Ne a b) (And (Not P) (Ne a b))
                               -/
  dite_ne_left_iff.trans <| by rw [exists_prop]
                               /-
                                 🎉 no goals
                               -/


theorem ite_ne_right_iff : ite P a b ≠ b ↔ P ∧ a ≠ b :=
                                /-
                                  α : Sort u_1
                                  P : Prop
                                  inst✝ : Decidable P
                                  a b : α
                                  ⊢ Iff (Exists fun h => Ne a b) (And P (Ne a b))
                                -/
  dite_ne_right_iff.trans <| by rw [exists_prop]
                                /-
                                  🎉 no goals
                                -/


protected theorem Ne.dite_eq_left_iff (h : ∀ h, a ≠ B h) : dite P (fun _ ↦ a) B = a ↔ P :=
  dite_eq_left_iff.trans ⟨fun H ↦ of_not_not fun h' ↦ h h' (H h').symm, fun h H ↦ (H h).elim⟩


protected theorem Ne.dite_eq_right_iff (h : ∀ h, A h ≠ b) : (dite P A fun _ ↦ b) = b ↔ ¬P :=
  dite_eq_right_iff.trans ⟨fun H h' ↦ h h' (H h'), fun h' H ↦ (h' H).elim⟩


protected theorem Ne.ite_eq_left_iff (h : a ≠ b) : ite P a b = a ↔ P :=
  Ne.dite_eq_left_iff fun _ ↦ h


protected theorem Ne.ite_eq_right_iff (h : a ≠ b) : ite P a b = b ↔ ¬P :=
  Ne.dite_eq_right_iff fun _ ↦ h


protected theorem Ne.dite_ne_left_iff (h : ∀ h, a ≠ B h) : dite P (fun _ ↦ a) B ≠ a ↔ ¬P :=
  dite_ne_left_iff.trans <| exists_iff_of_forall h


protected theorem Ne.dite_ne_right_iff (h : ∀ h, A h ≠ b) : (dite P A fun _ ↦ b) ≠ b ↔ P :=
  dite_ne_right_iff.trans <| exists_iff_of_forall h


protected theorem Ne.ite_ne_left_iff (h : a ≠ b) : ite P a b ≠ a ↔ ¬P :=
  Ne.dite_ne_left_iff fun _ ↦ h


protected theorem Ne.ite_ne_right_iff (h : a ≠ b) : ite P a b ≠ b ↔ P :=
  Ne.dite_ne_right_iff fun _ ↦ h


theorem dite_eq_or_eq : (∃ h, dite P A B = A h) ∨ ∃ h, dite P A B = B h :=
  if h : _ then .inl ⟨h, dif_pos h⟩ else .inr ⟨h, dif_neg h⟩


theorem ite_eq_or_eq : ite P a b = a ∨ ite P a b = b :=
  if h : _ then .inl (if_pos h) else .inr (if_neg h)


/-- A two-argument function applied to two `dite`s is a `dite` of that two-argument function
applied to each of the branches. -/
theorem apply_dite₂ {α β γ : Sort*} (f : α → β → γ) (P : Prop) [Decidable P]
    (a : P → α) (b : ¬P → α) (c : P → β) (d : ¬P → β) :
    f (dite P a b) (dite P c d) = dite P (fun h ↦ f (a h) (c h)) fun h ↦ f (b h) (d h) := by
  /-
    α : Sort u_3
    β : Sort u_4
    γ : Sort u_5
    f : α → β → γ
    P : Prop
    inst✝ : Decidable P
    a : P → α
    b : Not P → α
    c : P → β
    d : Not P → β
    ⊢ Eq (f (dite P a b) (dite P c d)) (dite P (fun h => f (a h) (c h)) fun h => f …
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases h : P <;> simp [h]
                     /-
                       🎉 no goals
                     -/


/-- A two-argument function applied to two `ite`s is a `ite` of that two-argument function
applied to each of the branches. -/
theorem apply_ite₂ {α β γ : Sort*} (f : α → β → γ) (P : Prop) [Decidable P] (a b : α) (c d : β) :
    f (ite P a b) (ite P c d) = ite P (f a c) (f b d) :=
  apply_dite₂ f P (fun _ ↦ a) (fun _ ↦ b) (fun _ ↦ c) fun _ ↦ d


/-- A 'dite' producing a `Pi` type `Π a, σ a`, applied to a value `a : α` is a `dite` that applies
either branch to `a`. -/
theorem dite_apply (f : P → ∀ a, σ a) (g : ¬P → ∀ a, σ a) (a : α) :
                                                                /-
                                                                  α : Sort u_1
                                                                  σ : α → Sort u_2
                                                                  P : Prop
                                                                  inst✝ : Decidable P
                                                                  f : P → (a : α) → σ a
                                                                  g : Not P → (a : α) → σ a
                                                                  a : α
                                                                  ⊢ Eq (dite P f g a) (dite P (fun h => f h a) fun h => g h a)
                                                                -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
    (dite P f g) a = dite P (fun h ↦ f h a) fun h ↦ g h a := by by_cases h : P <;> simp [h]
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- A 'ite' producing a `Pi` type `Π a, σ a`, applied to a value `a : α` is a `ite` that applies
either branch to `a`. -/
theorem ite_apply (f g : ∀ a, σ a) (a : α) : (ite P f g) a = ite P (f a) (g a) :=
  dite_apply P (fun _ ↦ f) (fun _ ↦ g) a


theorem ite_and : ite (P ∧ Q) a b = ite P (ite Q a b) b := by
  /-
    α : Sort u_1
    P Q : Prop
    inst✝¹ : Decidable P
    a b : α
    inst✝ : Decidable Q
    ⊢ Eq (ite (And P Q) a b) (ite P (ite Q a b) b)
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
  by_cases hp : P <;> by_cases hq : Q <;> simp [hp, hq]
                                          /-
                                            🎉 no goals
                                          -/


theorem ite_or : ite (P ∨ Q) a b = ite P a (ite Q a b) := by
  /-
    α : Sort u_1
    P Q : Prop
    inst✝¹ : Decidable P
    a b : α
    inst✝ : Decidable Q
    ⊢ Eq (ite (Or P Q) a b) (ite P a (ite Q a b))
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
  by_cases hp : P <;> by_cases hq : Q <;> simp [hp, hq]
                                          /-
                                            🎉 no goals
                                          -/


theorem dite_dite_comm {B : Q → α} {C : ¬P → ¬Q → α} (h : P → ¬Q) :
    (if p : P then A p else if q : Q then B q else C p q) =
     if q : Q then B q else if p : P then A p else C p q :=
  dite_eq_iff'.2 ⟨
               /-
                 α : Sort u_1
                 P Q : Prop
                 inst✝¹ : Decidable P
                 A : P → α
                 inst✝ : Decidable Q
                 B : Q → α
                 C : Not P → Not Q → α
                 h : P → Not Q
                 p : P
                 ⊢ Eq (A p) (dite Q (fun q => B q) fun q => dite P (fun p => A p) fun p => C p q)
               -/
    fun p ↦ by rw [dif_neg (h p), dif_pos p],
               /-
                 🎉 no goals
               -/
                /-
                  α : Sort u_1
                  P Q : Prop
                  inst✝¹ : Decidable P
                  A : P → α
                  inst✝ : Decidable Q
                  B : Q → α
                  C : Not P → Not Q → α
                  h : P → Not Q
                  np : Not P
                  ⊢ Eq (dite Q (fun q => B q) fun q => C np q) (dite Q (fun q => B q) fun q => d …
                -/
    fun np ↦ by congr; funext _; rw [dif_neg np]⟩
                                 /-
                                   🎉 no goals
                                 -/


theorem ite_ite_comm (h : P → ¬Q) :
    (if P then a else if Q then b else c) =
     if Q then b else if P then a else c :=
  dite_dite_comm P Q h


theorem ite_prop_iff_or : (if P then Q else R) ↔ (P ∧ Q ∨ ¬ P ∧ R) := by
  /-
    P Q R : Prop
    inst✝ : Decidable P
    ⊢ Iff (ite P Q R) (Or (And P Q) (And (Not P) R))
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases p : P <;> simp [p]
                     /-
                       🎉 no goals
                     -/


theorem dite_prop_iff_or {Q : P → Prop} {R : ¬P → Prop} :
    dite P Q R ↔ (∃ p, Q p) ∨ (∃ p, R p) := by
  /-
    P : Prop
    inst✝ : Decidable P
    Q : P → Prop
    R : Not P → Prop
    ⊢ Iff (dite P Q R) (Or (Exists fun p => Q p) (Exists fun p => R p))
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases h : P <;> simp [h, exists_prop_of_false, exists_prop_of_true]
                     /-
                       🎉 no goals
                     -/

-- TODO make this a simp lemma in a future PR

theorem ite_prop_iff_and : (if P then Q else R) ↔ ((P → Q) ∧ (¬ P → R)) := by
  /-
    P Q R : Prop
    inst✝ : Decidable P
    ⊢ Iff (ite P Q R) (And (P → Q) (Not P → R))
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases p : P <;> simp [p]
                     /-
                       🎉 no goals
                     -/


theorem dite_prop_iff_and {Q : P → Prop} {R : ¬P → Prop} :
    dite P Q R ↔ (∀ h, Q h) ∧ (∀ h, R h) := by
  /-
    P : Prop
    inst✝ : Decidable P
    Q : P → Prop
    R : Not P → Prop
    ⊢ Iff (dite P Q R) (And (∀ (h : P), Q h) (∀ (h : Not P), R h))
  -/
                     /-
                       🎉 no goals
                     -/
  by_cases h : P <;> simp [h, forall_prop_of_false, forall_prop_of_true]
                     /-
                       🎉 no goals
                     -/


theorem if_ctx_congr (h_c : P ↔ Q) (h_t : Q → x = u) (h_e : ¬Q → y = v) : ite P x y = ite Q u v :=
  match ‹Decidable P›, ‹Decidable Q› with
                                 /-
                                   α : Sort u_1
                                   P Q : Prop
                                   inst✝¹ : Decidable P
                                   inst✝ : Decidable Q
                                   x y u v : α
                                   h_c : Iff P Q
                                   h_t : Q → Eq x u
                                   h_e : Not Q → Eq y v
                                   h✝ : Not P
                                   h₂ : Not Q
                                   ⊢ Eq (ite P x y) (ite Q u v)
                                 -/
  | isFalse _,  isFalse h₂ => by simp_all
                                 /-
                                   🎉 no goals
                                 -/
                                 /-
                                   α : Sort u_1
                                   P Q : Prop
                                   inst✝¹ : Decidable P
                                   inst✝ : Decidable Q
                                   x y u v : α
                                   h_c : Iff P Q
                                   h_t : Q → Eq x u
                                   h_e : Not Q → Eq y v
                                   h✝ : P
                                   h₂ : Q
                                   ⊢ Eq (ite P x y) (ite Q u v)
                                 -/
  | isTrue _,   isTrue h₂  => by simp_all
                                 /-
                                   🎉 no goals
                                 -/
  | isFalse h₁, isTrue h₂  => absurd h₂ (Iff.mp (not_congr h_c) h₁)
  | isTrue h₁,  isFalse h₂ => absurd h₁ (Iff.mpr (not_congr h_c) h₂)


theorem if_congr (h_c : P ↔ Q) (h_t : x = u) (h_e : y = v) : ite P x y = ite Q u v :=
  if_ctx_congr h_c (fun _ ↦ h_t) (fun _ ↦ h_e)


theorem not_beq_of_ne {α : Type*} [BEq α] [LawfulBEq α] {a b : α} (ne : a ≠ b) : ¬(a == b) :=
  fun h => ne (eq_of_beq h)


theorem beq_eq_decide {α : Type*} [BEq α] [LawfulBEq α] {a b : α} : (a == b) = decide (a = b) := by
  /-
    α : Type u_1
    inst✝¹ : BEq α
    inst✝ : LawfulBEq α
    a b : α
    ⊢ Eq (BEq.beq a b) (Decidable.decide (Eq a b))
  -/
  rw [← beq_iff_eq (a := a) (b := b)]
  /-
    α : Type u_1
    inst✝¹ : BEq α
    inst✝ : LawfulBEq α
    a b : α
    ⊢ Eq (BEq.beq a b) (Decidable.decide (Eq (BEq.beq a b) Bool.true))
  -/
                   /-
                     🎉 no goals
                   -/
  cases a == b <;> simp
                   /-
                     🎉 no goals
                   -/


@[simp] lemma beq_eq_beq {α β : Type*} [BEq α] [LawfulBEq α] [BEq β] [LawfulBEq β] {a₁ a₂ : α}
                                                                      /-
                                                                        α : Type u_1
                                                                        β : Type u_2
                                                                        inst✝³ : BEq α
                                                                        inst✝² : LawfulBEq α
                                                                        inst✝¹ : BEq β
                                                                        inst✝ : LawfulBEq β
                                                                        a₁ a₂ : α
                                                                        b₁ b₂ : β
                                                                        ⊢ Iff (Eq (BEq.beq a₁ a₂) (BEq.beq b₁ b₂)) (Iff (Eq a₁ a₂) (Eq b₁ b₂))
                                                                      -/
    {b₁ b₂ : β} : (a₁ == a₂) = (b₁ == b₂) ↔ (a₁ = a₂ ↔ b₁ = b₂) := by rw [Bool.eq_iff_iff]; simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


@[ext]
theorem beq_ext {α : Type*} (inst1 : BEq α) (inst2 : BEq α)
    (h : ∀ x y, @BEq.beq _ inst1 x y = @BEq.beq _ inst2 x y) :
    inst1 = inst2 := by
  /-
    α : Type u_1
    inst1 inst2 : BEq α
    h : ∀ (x y : α), Eq (BEq.beq x y) (BEq.beq x y)
    ⊢ Eq inst1 inst2
  -/
  have ⟨beq1⟩ := inst1
  /-
    α : Type u_1
    inst1 inst2 : BEq α
    beq1 : α → α → Bool
    h : ∀ (x y : α), Eq (BEq.beq x y) (BEq.beq x y)
    ⊢ Eq { beq := beq1 } inst2
  -/
  have ⟨beq2⟩ := inst2
  /-
    α : Type u_1
    inst1 inst2 : BEq α
    beq1 beq2 : α → α → Bool
    h : ∀ (x y : α), Eq (BEq.beq x y) (BEq.beq x y)
    ⊢ Eq { beq := beq1 } { beq := beq2 }
  -/
  congr
  /-
    case e_beq
    α : Type u_1
    inst1 inst2 : BEq α
    beq1 beq2 : α → α → Bool
    h : ∀ (x y : α), Eq (BEq.beq x y) (BEq.beq x y)
    ⊢ Eq beq1 beq2
  -/
  funext x y
  /-
    case e_beq.h.h
    α : Type u_1
    inst1 inst2 : BEq α
    beq1 beq2 : α → α → Bool
    h : ∀ (x y : α), Eq (BEq.beq x y) (BEq.beq x y)
    x y : α
    ⊢ Eq (beq1 x y) (beq2 x y)
  -/
  exact h x y
  /-
    🎉 no goals
  -/


theorem lawful_beq_subsingleton {α : Type*} (inst1 : BEq α) (inst2 : BEq α)
    [@LawfulBEq α inst1] [@LawfulBEq α inst2] :
    inst1 = inst2 := by
  /-
    α : Type u_1
    inst1 inst2 : BEq α
    inst✝¹ : LawfulBEq α
    inst✝ : LawfulBEq α
    ⊢ Eq inst1 inst2
  -/
  apply beq_ext
  /-
    case h
    α : Type u_1
    inst1 inst2 : BEq α
    inst✝¹ : LawfulBEq α
    inst✝ : LawfulBEq α
    ⊢ ∀ (x y : α), Eq (BEq.beq x y) (BEq.beq x y)
  -/
  intro x y
  /-
    case h
    α : Type u_1
    inst1 inst2 : BEq α
    inst✝¹ : LawfulBEq α
    inst✝ : LawfulBEq α
    x y : α
    ⊢ Eq (BEq.beq x y) (BEq.beq x y)
  -/
  simp only [beq_eq_decide]
  /-
    🎉 no goals
  -/

