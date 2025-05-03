/-- `PFun α β`, or `α →. β`, is the type of partial functions from
  `α` to `β`. It is defined as `α → Part β`. -/
def PFun (α β : Type*) :=
  α → Part β


/-- `α →. β` is notation for the type `PFun α β` of partial functions from `α` to `β`. -/
infixr:25 " →. " => PFun


instance inhabited : Inhabited (α →. β) :=
  ⟨fun _ => Part.none⟩


/-- The domain of a partial function -/
def Dom (f : α →. β) : Set α :=
  { a | (f a).Dom }


@[simp]
                                                                      /-
                                                                        α : Type u_1
                                                                        β : Type u_2
                                                                        f : PFun α β
                                                                        x : α
                                                                        ⊢ Iff (Membership.mem f.Dom x) (Exists fun y => Membership.mem (f x) y)
                                                                      -/
theorem mem_dom (f : α →. β) (x : α) : x ∈ Dom f ↔ ∃ y, y ∈ f x := by simp [Dom, Part.dom_iff_mem]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp]
theorem dom_mk (p : α → Prop) (f : ∀ a, p a → β) : (PFun.Dom fun x => ⟨p x, f x⟩) = { x | p x } :=
  rfl


theorem dom_eq (f : α →. β) : Dom f = { x | ∃ y, y ∈ f x } :=
  Set.ext (mem_dom f)


/-- Evaluate a partial function -/
def fn (f : α →. β) (a : α) : Dom f a → β :=
  (f a).get


@[simp]
theorem fn_apply (f : α →. β) (a : α) : f.fn a = (f a).get :=
  rfl


/-- Evaluate a partial function to return an `Option` -/
def evalOpt (f : α →. β) [D : DecidablePred (· ∈ Dom f)] (x : α) : Option β :=
  @Part.toOption _ _ (D x)


/-- Partial function extensionality -/
theorem ext' {f g : α →. β} (H1 : ∀ a, a ∈ Dom f ↔ a ∈ Dom g) (H2 : ∀ a p q, f.fn a p = g.fn a q) :
    f = g :=
  funext fun a => Part.ext' (H1 a) (H2 a)


theorem ext {f g : α →. β} (H : ∀ a b, b ∈ f a ↔ b ∈ g a) : f = g :=
  funext fun a => Part.ext (H a)


/-- Turns a partial function into a function out of its domain. -/
def asSubtype (f : α →. β) (s : f.Dom) : β :=
  f.fn s s.2


/-- The type of partial functions `α →. β` is equivalent to
the type of pairs `(p : α → Prop, f : Subtype p → β)`. -/
def equivSubtype : (α →. β) ≃ Σp : α → Prop, Subtype p → β :=
  ⟨fun f => ⟨fun a => (f a).Dom, asSubtype f⟩, fun f x => ⟨f.1 x, fun h => f.2 ⟨x, h⟩⟩, fun _ =>
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   γ : Type u_3
                                                   δ : Type u_4
                                                   ε : Type u_5
                                                   ι : Type u_6
                                                   x✝ : Sigma fun p => Subtype p → β
                                                   p : α → Prop
                                                   f : Subtype p → β
                                                   ⊢ Eq ((fun f => ⟨fun a => (f a).Dom, f.asSubtype⟩) ((fun f x => { Dom := f.fst …
                                                 -/
    funext fun _ => Part.eta _, fun ⟨p, f⟩ => by dsimp; congr⟩
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem asSubtype_eq_of_mem {f : α →. β} {x : α} {y : β} (fxy : y ∈ f x) (domx : x ∈ f.Dom) :
    f.asSubtype ⟨x, domx⟩ = y :=
  Part.mem_unique (Part.get_mem _) fxy


/-- Turn a total function into a partial function. -/
@[coe]
protected def lift (f : α → β) : α →. β := fun a => Part.some (f a)


instance coe : Coe (α → β) (α →. β) :=
  ⟨PFun.lift⟩


@[simp]
theorem coe_val (f : α → β) (a : α) : (f : α →. β) a = Part.some (f a) :=
  rfl


@[simp]
theorem dom_coe (f : α → β) : (f : α →. β).Dom = Set.univ :=
  rfl


theorem lift_injective : Injective (PFun.lift : (α → β) → α →. β) := fun _ _ h =>
  funext fun a => Part.some_injective <| congr_fun h a


/-- Graph of a partial function `f` as the set of pairs `(x, f x)` where `x` is in the domain of
`f`. -/
def graph (f : α →. β) : Set (α × β) :=
  { p | p.2 ∈ f p.1 }


/-- Graph of a partial function as a relation. `x` and `y` are related iff `f x` is defined and
"equals" `y`. -/
def graph' (f : α →. β) : Rel α β := fun x y => y ∈ f x


/-- The range of a partial function is the set of values
  `f x` where `x` is in the domain of `f`. -/
def ran (f : α →. β) : Set β :=
  { b | ∃ a, b ∈ f a }


/-- Restrict a partial function to a smaller domain. -/
def restrict (f : α →. β) {p : Set α} (H : p ⊆ f.Dom) : α →. β := fun x =>
  (f x).restrict (x ∈ p) (@H x)


@[simp]
theorem mem_restrict {f : α →. β} {s : Set α} (h : s ⊆ f.Dom) (a : α) (b : β) :
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 f : PFun α β
                                                 s : Set α
                                                 h : HasSubset.Subset s f.Dom
                                                 a : α
                                                 b : β
                                                 ⊢ Iff (Membership.mem (f.restrict h a) b) (And (Membership.mem s a) (Membershi …
                                               -/
    b ∈ f.restrict h a ↔ a ∈ s ∧ b ∈ f a := by simp [restrict]
                                               /-
                                                 🎉 no goals
                                               -/


/-- Turns a function into a partial function with a prescribed domain. -/
def res (f : α → β) (s : Set α) : α →. β :=
  (PFun.lift f).restrict s.subset_univ


theorem mem_res (f : α → β) (s : Set α) (a : α) (b : β) : b ∈ res f s a ↔ a ∈ s ∧ f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    a : α
    b : β
    ⊢ Iff (Membership.mem (PFun.res f s a) b) (And (Membership.mem s a) (Eq (f a)  …
  -/
  simp [res, @eq_comm _ b]
  /-
    🎉 no goals
  -/


theorem res_univ (f : α → β) : PFun.res f Set.univ = f :=
  rfl


theorem dom_iff_graph (f : α →. β) (x : α) : x ∈ f.Dom ↔ ∃ y, (x, y) ∈ f.graph :=
  Part.dom_iff_mem


theorem lift_graph {f : α → β} {a b} : (a, b) ∈ (f : α →. β).graph ↔ f a = b :=
                                          /-
                                            α : Type u_1
                                            β : Type u_2
                                            f : α → β
                                            a : α
                                            b : β
                                            ⊢ Iff (Exists fun x => Eq (f a) b) (Eq (f a) b)
                                          -/
  show (∃ _ : True, f a = b) ↔ f a = b by simp
                                          /-
                                            🎉 no goals
                                          -/


/-- The monad `pure` function, the total constant `x` function -/
protected def pure (x : β) : α →. β := fun _ => Part.some x


/-- The monad `bind` function, pointwise `Part.bind` -/
def bind (f : α →. β) (g : β → α →. γ) : α →. γ := fun a => (f a).bind fun b => g b a


@[simp]
theorem bind_apply (f : α →. β) (g : β → α →. γ) (a : α) : f.bind g a = (f a).bind fun b => g b a :=
  rfl


/-- The monad `map` function, pointwise `Part.map` -/
def map (f : β → γ) (g : α →. β) : α →. γ := fun a => (g a).map f


instance monad : Monad (PFun α) where
  pure := PFun.pure
  bind := PFun.bind
  map := PFun.map


                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 γ : Type u_3
                                                 δ : Type u_4
                                                 ε : Type u_5
                                                 ι : Type u_6
                                                 ⊢ ∀ {α_1 β : Type u_7} (x : α_1) (y : PFun α β), Eq (Functor.mapConst x y) (Fu …
                                               -/
                                               /-
                                                 🎉 no goals
                                               -/
                         /-
                           α : Type u_1
                           β : Type u_2
                           γ : Type u_3
                           δ : Type u_4
                           ε : Type u_5
                           ι : Type u_6
                           α✝ : Type u_7
                           f : PFun α α✝
                           ⊢ Eq (Functor.map id f) f
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
instance lawfulMonad : LawfulMonad (PFun α) := LawfulMonad.mk'
                                               /-
                                                 🎉 no goals
                                               -/
  (bind_pure_comp := fun _ _ => funext fun _ => Part.bind_some_eq_map _ _)
  (id_map := fun f => by funext a; dsimp [Functor.map, PFun.map]; cases f a; rfl)
  (pure_bind := fun x f => funext fun _ => Part.bind_some _ (f x))
  (bind_assoc := fun f g k => funext fun a => (f a).bind_assoc (fun b => g b a) fun b => k b a)


theorem pure_defined (p : Set α) (x : β) : p ⊆ (@PFun.pure α _ x).Dom :=
  p.subset_univ


theorem bind_defined {α β γ} (p : Set α) {f : α →. β} {g : β → α →. γ} (H1 : p ⊆ f.Dom)
    (H2 : ∀ x, p ⊆ (g x).Dom) : p ⊆ (f >>= g).Dom := fun a ha =>
  (⟨H1 ha, H2 _ ha⟩ : (f >>= g).Dom a)


/-- First return map. Transforms a partial function `f : α →. β ⊕ α` into the partial function
`α →. β` which sends `a : α` to the first value in `β` it hits by iterating `f`, if such a value
exists. By abusing notation to illustrate, either `f a` is in the `β` part of `β ⊕ α` (in which
case `f.fix a` returns `f a`), or it is undefined (in which case `f.fix a` is undefined as well), or
it is in the `α` part of `β ⊕ α` (in which case we repeat the procedure, so `f.fix a` will return
`f.fix (f a)`). -/
def fix (f : α →. β ⊕ α) : α →. β := fun a =>
  Part.assert (Acc (fun x y => Sum.inr x ∈ f y) a) fun h =>
    WellFounded.fixF
      (fun a IH =>
        Part.assert (f a).Dom fun hf =>
          match e : (f a).get hf with
          | Sum.inl b => Part.some b
          | Sum.inr a' => IH a' ⟨hf, e⟩)
      a h


theorem dom_of_mem_fix {f : α →. β ⊕ α} {a : α} {b : β} (h : b ∈ f.fix a) : (f a).Dom := by
  /-
    α : Type u_1
    β : Type u_2
    f : PFun α (Sum β α)
    a : α
    b : β
    h : Membership.mem (f.fix a) b
    ⊢ (f a).Dom
  -/
  let ⟨h₁, h₂⟩ := Part.mem_assert_iff.1 h
  /-
    α : Type u_1
    β : Type u_2
    f : PFun α (Sum β α)
    a : α
    b : β
    h : Membership.mem (f.fix a) b
    h₁ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
    h₂ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
    ⊢ (f a).Dom
  -/
  rw [WellFounded.fixFEq] at h₂; exact h₂.fst.fst
                                 /-
                                   🎉 no goals
                                 -/


theorem mem_fix_iff {f : α →. β ⊕ α} {a : α} {b : β} :
    b ∈ f.fix a ↔ Sum.inl b ∈ f a ∨ ∃ a', Sum.inr a' ∈ f a ∧ b ∈ f.fix a' :=
  ⟨fun h => by
    /-
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Membership.mem (f.fix a) b
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    let ⟨h₁, h₂⟩ := Part.mem_assert_iff.1 h
    /-
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Membership.mem (f.fix a) b
      h₁ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
      h₂ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    rw [WellFounded.fixFEq] at h₂
    /-
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Membership.mem (f.fix a) b
      h₁ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
      h₂ : Membership.mem (Part.assert (f a).Dom fun hf => PFun.fix.match_1 (fun x = …
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    simp only [Part.mem_assert_iff] at h₂
    /-
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Membership.mem (f.fix a) b
      h₁ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
      h₂ : Exists fun h => Membership.mem (PFun.fix.match_1 (fun x => Part β) ((f a) …
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    cases' h₂ with h₂ h₃
    /-
      case intro
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Membership.mem (f.fix a) b
      h₁ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
      h₂ : (f a).Dom
      h₃ : Membership.mem (PFun.fix.match_1 (fun x => Part β) ((f a).get h₂) (fun b  …
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    split at h₃
    /-
      case intro.h_1
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Membership.mem (f.fix a) b
      h₁ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
      h₂ : (f a).Dom
      b✝ : β
      heq✝ : Eq ((f a).get h₂) (Sum.inl b✝)
      h₃ : Membership.mem (Part.some b✝) b
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    next e => simp only [Part.mem_some_iff] at h₃; subst b; exact Or.inl ⟨h₂, e⟩
    /-
      case intro.h_2
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Membership.mem (f.fix a) b
      h₁ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
      h₂ : (f a).Dom
      a'✝ : α
      heq✝ : Eq ((f a).get h₂) (Sum.inr a'✝)
      h₃ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    next e => exact Or.inr ⟨_, ⟨_, e⟩, Part.mem_assert _ h₃⟩,
    /-
      🎉 no goals
    -/
   fun h => by
    /-
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.me …
      ⊢ Membership.mem (f.fix a) b
    -/
    simp only [fix, Part.mem_assert_iff]
    /-
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a : α
      b : β
      h : Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.me …
      ⊢ Exists fun h => Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f …
    -/
    rcases h with (⟨h₁, h₂⟩ | ⟨a', h, h₃⟩)
      /-
        case inl.intro
        α : Type u_1
        β : Type u_2
        f : PFun α (Sum β α)
        a : α
        b : β
        h₁ : (f a).Dom
        h₂ : Eq ((f a).get h₁) (Sum.inl b)
        ⊢ Exists fun h => Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f …
      -/
    · refine ⟨⟨_, fun y h' => ?_⟩, ?_⟩
        /-
          case inl.intro.refine_1
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          h₁ : (f a).Dom
          h₂ : Eq ((f a).get h₁) (Sum.inl b)
          y : α
          h' : Membership.mem (f a) (Sum.inr y)
          ⊢ Acc (fun x y => Membership.mem (f y) (Sum.inr x)) y
        -/
      · injection Part.mem_unique ⟨h₁, h₂⟩ h'
        /-
          🎉 no goals
        -/
        /-
          case inl.intro.refine_2
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          h₁ : (f a).Dom
          h₂ : Eq ((f a).get h₁) (Sum.inl b)
          ⊢ Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun hf = …
        -/
      · rw [WellFounded.fixFEq]
        -- Porting note: used to be simp [h₁, h₂]
        /-
          case inl.intro.refine_2
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          h₁ : (f a).Dom
          h₂ : Eq ((f a).get h₁) (Sum.inl b)
          ⊢ Membership.mem (Part.assert (f a).Dom fun hf => PFun.fix.match_1 (fun x => P …
        -/
        apply Part.mem_assert h₁
        /-
          case inl.intro.refine_2
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          h₁ : (f a).Dom
          h₂ : Eq ((f a).get h₁) (Sum.inl b)
          ⊢ Membership.mem (PFun.fix.match_1 (fun x => Part β) ((f a).get h₁) (fun b e = …
        -/
        split
        next e =>
          injection h₂.symm.trans e with h; simp [h]
        next e =>
          injection h₂.symm.trans e
      /-
        case inr.intro.intro
        α : Type u_1
        β : Type u_2
        f : PFun α (Sum β α)
        a : α
        b : β
        a' : α
        h : Membership.mem (f a) (Sum.inr a')
        h₃ : Membership.mem (f.fix a') b
        ⊢ Exists fun h => Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f …
      -/
    · simp only [fix, Part.mem_assert_iff] at h₃
      /-
        case inr.intro.intro
        α : Type u_1
        β : Type u_2
        f : PFun α (Sum β α)
        a : α
        b : β
        a' : α
        h : Membership.mem (f a) (Sum.inr a')
        h₃ : Exists fun h => Membership.mem (WellFounded.fixF (fun a IH => Part.assert …
        ⊢ Exists fun h => Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f …
      -/
      cases' h₃ with h₃ h₄
      /-
        case inr.intro.intro.intro
        α : Type u_1
        β : Type u_2
        f : PFun α (Sum β α)
        a : α
        b : β
        a' : α
        h : Membership.mem (f a) (Sum.inr a')
        h₃ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a'
        h₄ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
        ⊢ Exists fun h => Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f …
      -/
      refine ⟨⟨_, fun y h' => ?_⟩, ?_⟩
        /-
          case inr.intro.intro.intro.refine_1
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          a' : α
          h : Membership.mem (f a) (Sum.inr a')
          h₃ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a'
          h₄ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
          y : α
          h' : Membership.mem (f a) (Sum.inr y)
          ⊢ Acc (fun x y => Membership.mem (f y) (Sum.inr x)) y
        -/
      · injection Part.mem_unique h h' with e
        /-
          case inr.intro.intro.intro.refine_1
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          a' : α
          h : Membership.mem (f a) (Sum.inr a')
          h₃ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a'
          h₄ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
          y : α
          h' : Membership.mem (f a) (Sum.inr y)
          e : Eq a' y
          ⊢ Acc (fun x y => Membership.mem (f y) (Sum.inr x)) y
        -/
        exact e ▸ h₃
        /-
          🎉 no goals
        -/
        /-
          case inr.intro.intro.intro.refine_2
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          a' : α
          h : Membership.mem (f a) (Sum.inr a')
          h₃ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a'
          h₄ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
          ⊢ Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun hf = …
        -/
      · cases' h with h₁ h₂
        /-
          case inr.intro.intro.intro.refine_2.intro
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          a' : α
          h₃ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a'
          h₄ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
          h₁ : (f a).Dom
          h₂ : Eq ((f a).get h₁) (Sum.inr a')
          ⊢ Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun hf = …
        -/
        rw [WellFounded.fixFEq]
        -- Porting note: used to be simp [h₁, h₂, h₄]
        /-
          case inr.intro.intro.intro.refine_2.intro
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          a' : α
          h₃ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a'
          h₄ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
          h₁ : (f a).Dom
          h₂ : Eq ((f a).get h₁) (Sum.inr a')
          ⊢ Membership.mem (Part.assert (f a).Dom fun hf => PFun.fix.match_1 (fun x => P …
        -/
        apply Part.mem_assert h₁
        /-
          case inr.intro.intro.intro.refine_2.intro
          α : Type u_1
          β : Type u_2
          f : PFun α (Sum β α)
          a : α
          b : β
          a' : α
          h₃ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a'
          h₄ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
          h₁ : (f a).Dom
          h₂ : Eq ((f a).get h₁) (Sum.inr a')
          ⊢ Membership.mem (PFun.fix.match_1 (fun x => Part β) ((f a).get h₁) (fun b e = …
        -/
        split
        next e =>
          injection h₂.symm.trans e
        next e =>
          injection h₂.symm.trans e; subst a'; exact h₄⟩


/-- If advancing one step from `a` leads to `b : β`, then `f.fix a = b` -/
theorem fix_stop {f : α →. β ⊕ α} {b : β} {a : α} (hb : Sum.inl b ∈ f a) : b ∈ f.fix a := by
  /-
    α : Type u_1
    β : Type u_2
    f : PFun α (Sum β α)
    b : β
    a : α
    hb : Membership.mem (f a) (Sum.inl b)
    ⊢ Membership.mem (f.fix a) b
  -/
  rw [PFun.mem_fix_iff]
  /-
    α : Type u_1
    β : Type u_2
    f : PFun α (Sum β α)
    b : β
    a : α
    hb : Membership.mem (f a) (Sum.inl b)
    ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
  -/
  exact Or.inl hb
  /-
    🎉 no goals
  -/


/-- If advancing one step from `a` on `f` leads to `a' : α`, then `f.fix a = f.fix a'` -/
theorem fix_fwd_eq {f : α →. β ⊕ α} {a a' : α} (ha' : Sum.inr a' ∈ f a) : f.fix a = f.fix a' := by
  /-
    α : Type u_1
    β : Type u_2
    f : PFun α (Sum β α)
    a a' : α
    ha' : Membership.mem (f a) (Sum.inr a')
    ⊢ Eq (f.fix a) (f.fix a')
  -/
  ext b; constructor
    /-
      case H.mp
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a a' : α
      ha' : Membership.mem (f a) (Sum.inr a')
      b : β
      ⊢ Membership.mem (f.fix a) b → Membership.mem (f.fix a') b
    -/
  · intro h
    /-
      case H.mp
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a a' : α
      ha' : Membership.mem (f a) (Sum.inr a')
      b : β
      h : Membership.mem (f.fix a) b
      ⊢ Membership.mem (f.fix a') b
    -/
                                                   /-
                                                     🎉 no goals
                                                   -/
    obtain h' | ⟨a, h', e'⟩ := mem_fix_iff.1 h <;> cases Part.mem_unique ha' h'
    /-
      case H.mp.inr.intro.intro.refl
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a a' : α
      ha' : Membership.mem (f a) (Sum.inr a')
      b : β
      h : Membership.mem (f.fix a) b
      h' : Membership.mem (f a) (Sum.inr a')
      e' : Membership.mem (f.fix a') b
      ⊢ Membership.mem (f.fix a') b
    -/
    exact e'
    /-
      🎉 no goals
    -/
    /-
      case H.mpr
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a a' : α
      ha' : Membership.mem (f a) (Sum.inr a')
      b : β
      ⊢ Membership.mem (f.fix a') b → Membership.mem (f.fix a) b
    -/
  · intro h
    /-
      case H.mpr
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a a' : α
      ha' : Membership.mem (f a) (Sum.inr a')
      b : β
      h : Membership.mem (f.fix a') b
      ⊢ Membership.mem (f.fix a) b
    -/
    rw [PFun.mem_fix_iff]
    /-
      case H.mpr
      α : Type u_1
      β : Type u_2
      f : PFun α (Sum β α)
      a a' : α
      ha' : Membership.mem (f a) (Sum.inr a')
      b : β
      h : Membership.mem (f.fix a') b
      ⊢ Or (Membership.mem (f a) (Sum.inl b)) (Exists fun a' => And (Membership.mem  …
    -/
    exact Or.inr ⟨a', ha', h⟩
    /-
      🎉 no goals
    -/


theorem fix_fwd {f : α →. β ⊕ α} {b : β} {a a' : α} (hb : b ∈ f.fix a) (ha' : Sum.inr a' ∈ f a) :
                       /-
                         α : Type u_1
                         β : Type u_2
                         f : PFun α (Sum β α)
                         b : β
                         a a' : α
                         hb : Membership.mem (f.fix a) b
                         ha' : Membership.mem (f a) (Sum.inr a')
                         ⊢ Membership.mem (f.fix a') b
                       -/
    b ∈ f.fix a' := by rwa [← fix_fwd_eq ha']
                       /-
                         🎉 no goals
                       -/


/-- A recursion principle for `PFun.fix`. -/
@[elab_as_elim]
def fixInduction {C : α → Sort*} {f : α →. β ⊕ α} {b : β} {a : α} (h : b ∈ f.fix a)
    (H : ∀ a', b ∈ f.fix a' → (∀ a'', Sum.inr a'' ∈ f a' → C a'') → C a') : C a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    ⊢ C a
  -/
  have h₂ := (Part.mem_assert_iff.1 h).snd
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    h₂ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
    ⊢ C a
  -/
  generalize_proofs at h₂
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    pf✝¹ : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr a') → M …
    pf✝ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
    h₂ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
    ⊢ C a
  -/
  clear h
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    pf✝¹ : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr a') → M …
    pf✝ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
    h₂ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
    ⊢ C a
  -/
  induction ‹Acc _ _› with | intro a ha IH => _
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a✝ : α
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    pf✝ : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr a') → Me …
    a : α
    ha : ∀ (y : α), Membership.mem (f a) (Sum.inr y) → Acc (fun x y => Membership. …
    IH : (y : α) → (a : Membership.mem (f a) (Sum.inr y)) → Membership.mem (WellFo …
    h₂ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
    ⊢ C a
  -/
  have h : b ∈ f.fix a := Part.mem_assert_iff.2 ⟨⟨a, ha⟩, h₂⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a✝ : α
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    pf✝ : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr a') → Me …
    a : α
    ha : ∀ (y : α), Membership.mem (f a) (Sum.inr y) → Acc (fun x y => Membership. …
    IH : (y : α) → (a : Membership.mem (f a) (Sum.inr y)) → Membership.mem (WellFo …
    h₂ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun h …
    h : Membership.mem (f.fix a) b
    ⊢ C a
  -/
  exact H a h fun a' fa' => IH a' fa' (Part.mem_assert_iff.1 (fix_fwd h fa')).snd
  /-
    🎉 no goals
  -/


theorem fixInduction_spec {C : α → Sort*} {f : α →. β ⊕ α} {b : β} {a : α} (h : b ∈ f.fix a)
    (H : ∀ a', b ∈ f.fix a' → (∀ a'', Sum.inr a'' ∈ f a' → C a'') → C a') :
    @fixInduction _ _ C _ _ _ h H = H a h fun _ h' => fixInduction (fix_fwd h h') H := by
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    ⊢ Eq (PFun.fixInduction h H) (H a h fun x h' => PFun.fixInduction ⋯ H)
  -/
  unfold fixInduction
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    ⊢ Eq (letFun ⋯ fun h₂ => (fun pf pf_1 h₂ => Acc.rec (motive := fun {a} pf_2 => …
  -/
  generalize_proofs
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    pf✝⁷ : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr a') → M …
    pf✝⁶ : Acc (fun x y => Membership.mem (f y) (Sum.inr x)) a
    pf✝⁵ : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun …
    pf✝⁴ : ∀ (a : α), (∀ (y : α), Membership.mem (f a) (Sum.inr y) → Acc (fun x y  …
    pf✝³ : ∀ (pf : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr …
    pf✝² : ∀ (pf : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr …
    pf✝¹ : ∀ (x : α), Membership.mem (f a) (Sum.inr x) → Acc (fun x y => Membershi …
    pf✝ : ∀ (x : α) (h' : Membership.mem (f a) (Sum.inr x)), Membership.mem (WellF …
    ⊢ Eq (letFun pf✝⁵ fun h₂ => (fun pf pf_1 h₂ => Acc.rec (motive := fun {a} pf_2 …
  -/
  induction ‹Acc _ _›
  /-
    case intro
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    H : (a' : α) → Membership.mem (f.fix a') b → ((a'' : α) → Membership.mem (f a' …
    pf✝⁶ : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr a') → M …
    pf✝⁵ : ∀ (a : α), (∀ (y : α), Membership.mem (f a) (Sum.inr y) → Acc (fun x y  …
    pf✝⁴ : ∀ (pf : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr …
    pf✝³ : ∀ (pf : ∀ (a : α) (hf : (f a).Dom) (a' : α), Eq ((f a).get hf) (Sum.inr …
    x✝ : α
    h✝ : ∀ (y : α), Membership.mem (f x✝) (Sum.inr y) → Acc (fun x y => Membership …
    h_ih✝ : ∀ (y : α) (a : Membership.mem (f x✝) (Sum.inr y)) (h : Membership.mem  …
    h : Membership.mem (f.fix x✝) b
    pf✝² : Membership.mem (WellFounded.fixF (fun a IH => Part.assert (f a).Dom fun …
    pf✝¹ : ∀ (x : α), Membership.mem (f x✝) (Sum.inr x) → Acc (fun x y => Membersh …
    pf✝ : ∀ (x : α) (h' : Membership.mem (f x✝) (Sum.inr x)), Membership.mem (Well …
    ⊢ Eq (letFun pf✝² fun h₂ => (fun pf pf_1 h₂ => Acc.rec (motive := fun {a} pf_2 …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Another induction lemma for `b ∈ f.fix a` which allows one to prove a predicate `P` holds for
`a` given that `f a` inherits `P` from `a` and `P` holds for preimages of `b`.
-/
@[elab_as_elim]
def fixInduction' {C : α → Sort*} {f : α →. β ⊕ α} {b : β} {a : α}
    (h : b ∈ f.fix a) (hbase : ∀ a_final : α, Sum.inl b ∈ f a_final → C a_final)
    (hind : ∀ a₀ a₁ : α, b ∈ f.fix a₁ → Sum.inr a₁ ∈ f a₀ → C a₁ → C a₀) : C a := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    ⊢ C a
  -/
  refine fixInduction h fun a' h ih => ?_
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ε : Type u_5
    ι : Type u_6
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h✝ : Membership.mem (f.fix a) b
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    a' : α
    h : Membership.mem (f.fix a') b
    ih : (a'' : α) → Membership.mem (f a') (Sum.inr a'') → C a''
    ⊢ C a'
  -/
  rcases e : (f a').get (dom_of_mem_fix h) with b' | a'' <;> replace e : _ ∈ f a' := ⟨_, e⟩
    /-
      case inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      ι : Type u_6
      C : α → Sort u_7
      f : PFun α (Sum β α)
      b : β
      a : α
      h✝ : Membership.mem (f.fix a) b
      hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
      hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
      a' : α
      h : Membership.mem (f.fix a') b
      ih : (a'' : α) → Membership.mem (f a') (Sum.inr a'') → C a''
      b' : β
      e : Membership.mem (f a') (Sum.inl b')
      ⊢ C a'
    -/
  · apply hbase
    /-
      case inl.a
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      ι : Type u_6
      C : α → Sort u_7
      f : PFun α (Sum β α)
      b : β
      a : α
      h✝ : Membership.mem (f.fix a) b
      hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
      hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
      a' : α
      h : Membership.mem (f.fix a') b
      ih : (a'' : α) → Membership.mem (f a') (Sum.inr a'') → C a''
      b' : β
      e : Membership.mem (f a') (Sum.inl b')
      ⊢ Membership.mem (f a') (Sum.inl b)
    -/
    convert e
    /-
      case h.e'_5.h.e'_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      ι : Type u_6
      C : α → Sort u_7
      f : PFun α (Sum β α)
      b : β
      a : α
      h✝ : Membership.mem (f.fix a) b
      hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
      hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
      a' : α
      h : Membership.mem (f.fix a') b
      ih : (a'' : α) → Membership.mem (f a') (Sum.inr a'') → C a''
      b' : β
      e : Membership.mem (f a') (Sum.inl b')
      ⊢ Eq b b'
    -/
    exact Part.mem_unique h (fix_stop e)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ε : Type u_5
      ι : Type u_6
      C : α → Sort u_7
      f : PFun α (Sum β α)
      b : β
      a : α
      h✝ : Membership.mem (f.fix a) b
      hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
      hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
      a' : α
      h : Membership.mem (f.fix a') b
      ih : (a'' : α) → Membership.mem (f a') (Sum.inr a'') → C a''
      a'' : α
      e : Membership.mem (f a') (Sum.inr a'')
      ⊢ C a'
    -/
  · exact hind _ _ (fix_fwd h e) e (ih _ e)
    /-
      🎉 no goals
    -/


theorem fixInduction'_stop {C : α → Sort*} {f : α →. β ⊕ α} {b : β} {a : α} (h : b ∈ f.fix a)
    (fa : Sum.inl b ∈ f a) (hbase : ∀ a_final : α, Sum.inl b ∈ f a_final → C a_final)
    (hind : ∀ a₀ a₁ : α, b ∈ f.fix a₁ → Sum.inr a₁ ∈ f a₀ → C a₁ → C a₀) :
    @fixInduction' _ _ C _ _ _ h hbase hind = hbase a fa := by
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    fa : Membership.mem (f a) (Sum.inl b)
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    ⊢ Eq (PFun.fixInduction' h hbase hind) (hbase a fa)
  -/
  unfold fixInduction'
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    fa : Membership.mem (f a) (Sum.inl b)
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    ⊢ Eq (PFun.fixInduction h fun a' h ih => Sum.casesOn (motive := fun x => Eq (( …
  -/
  rw [fixInduction_spec]
  -- Porting note: the explicit motive required because `simp` behaves differently
  refine Eq.rec (motive := fun x e ↦
      Sum.casesOn x ?_ ?_ (Eq.trans (Part.get_eq_of_mem fa (dom_of_mem_fix h)) e) = hbase a fa) ?_
    (Part.get_eq_of_mem fa (dom_of_mem_fix h)).symm
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a : α
    h : Membership.mem (f.fix a) b
    fa : Membership.mem (f a) (Sum.inl b)
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    ⊢ (fun x e => Eq (Sum.casesOn (motive := fun x => Eq ((f a).get ⋯) x → C a) x  …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem fixInduction'_fwd {C : α → Sort*} {f : α →. β ⊕ α} {b : β} {a a' : α} (h : b ∈ f.fix a)
    (h' : b ∈ f.fix a') (fa : Sum.inr a' ∈ f a)
    (hbase : ∀ a_final : α, Sum.inl b ∈ f a_final → C a_final)
    (hind : ∀ a₀ a₁ : α, b ∈ f.fix a₁ → Sum.inr a₁ ∈ f a₀ → C a₁ → C a₀) :
    @fixInduction' _ _ C _ _ _ h hbase hind = hind a a' h' fa (fixInduction' h' hbase hind) := by
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a a' : α
    h : Membership.mem (f.fix a) b
    h' : Membership.mem (f.fix a') b
    fa : Membership.mem (f a) (Sum.inr a')
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    ⊢ Eq (PFun.fixInduction' h hbase hind) (hind a a' h' fa (PFun.fixInduction' h' …
  -/
  unfold fixInduction'
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a a' : α
    h : Membership.mem (f.fix a) b
    h' : Membership.mem (f.fix a') b
    fa : Membership.mem (f a) (Sum.inr a')
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    ⊢ Eq (PFun.fixInduction h fun a' h ih => Sum.casesOn (motive := fun x => Eq (( …
  -/
  rw [fixInduction_spec]
  -- Porting note: the explicit motive required because `simp` behaves differently
  refine Eq.rec (motive := fun x e =>
      Sum.casesOn (motive := fun y => (f a).get (dom_of_mem_fix h) = y → C a) x ?_ ?_
      (Eq.trans (Part.get_eq_of_mem fa (dom_of_mem_fix h)) e) = _) ?_
    (Part.get_eq_of_mem fa (dom_of_mem_fix h)).symm
  /-
    α : Type u_1
    β : Type u_2
    C : α → Sort u_7
    f : PFun α (Sum β α)
    b : β
    a a' : α
    h : Membership.mem (f.fix a) b
    h' : Membership.mem (f.fix a') b
    fa : Membership.mem (f a) (Sum.inr a')
    hbase : (a_final : α) → Membership.mem (f a_final) (Sum.inl b) → C a_final
    hind : (a₀ a₁ : α) → Membership.mem (f.fix a₁) b → Membership.mem (f a₀) (Sum. …
    ⊢ (fun x e => Eq (Sum.casesOn (motive := fun y => Eq ((f a).get ⋯) y → C a) x  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Image of a set under a partial function. -/
def image (s : Set α) : Set β :=
  f.graph'.image s


theorem image_def (s : Set α) : f.image s = { y | ∃ x ∈ s, y ∈ f x } :=
  rfl


theorem mem_image (y : β) (s : Set α) : y ∈ f.image s ↔ ∃ x ∈ s, y ∈ f x :=
  Iff.rfl


theorem image_mono {s t : Set α} (h : s ⊆ t) : f.image s ⊆ f.image t :=
  Rel.image_mono _ h


theorem image_inter (s t : Set α) : f.image (s ∩ t) ⊆ f.image s ∩ f.image t :=
  Rel.image_inter _ s t


theorem image_union (s t : Set α) : f.image (s ∪ t) = f.image s ∪ f.image t :=
  Rel.image_union _ s t


/-- Preimage of a set under a partial function. -/
def preimage (s : Set β) : Set α :=
  Rel.image (fun x y => x ∈ f y) s


theorem Preimage_def (s : Set β) : f.preimage s = { x | ∃ y ∈ s, y ∈ f x } :=
  rfl


@[simp]
theorem mem_preimage (s : Set β) (x : α) : x ∈ f.preimage s ↔ ∃ y ∈ s, y ∈ f x :=
  Iff.rfl


theorem preimage_subset_dom (s : Set β) : f.preimage s ⊆ f.Dom := fun _ ⟨y, _, fxy⟩ =>
  Part.dom_iff_mem.mpr ⟨y, fxy⟩


theorem preimage_mono {s t : Set β} (h : s ⊆ t) : f.preimage s ⊆ f.preimage t :=
  Rel.preimage_mono _ h


theorem preimage_inter (s t : Set β) : f.preimage (s ∩ t) ⊆ f.preimage s ∩ f.preimage t :=
  Rel.preimage_inter _ s t


theorem preimage_union (s t : Set β) : f.preimage (s ∪ t) = f.preimage s ∪ f.preimage t :=
  Rel.preimage_union _ s t


                                                          /-
                                                            α : Type u_1
                                                            β : Type u_2
                                                            f : PFun α β
                                                            ⊢ Eq (f.preimage Set.univ) f.Dom
                                                          -/
theorem preimage_univ : f.preimage Set.univ = f.Dom := by ext; simp [mem_preimage, mem_dom]
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                                                       /-
                                                                                         α : Type u_1
                                                                                         β : Type u_2
                                                                                         f : α → β
                                                                                         s : Set β
                                                                                         ⊢ Eq ((↑f).preimage s) (Set.preimage f s)
                                                                                       -/
theorem coe_preimage (f : α → β) (s : Set β) : (f : α →. β).preimage s = f ⁻¹' s := by ext; simp
                                                                                            /-
                                                                                              🎉 no goals
                                                                                            -/


/-- Core of a set `s : Set β` with respect to a partial function `f : α →. β`. Set of all `a : α`
such that `f a ∈ s`, if `f a` is defined. -/
def core (s : Set β) : Set α :=
  f.graph'.core s


theorem core_def (s : Set β) : f.core s = { x | ∀ y, y ∈ f x → y ∈ s } :=
  rfl


@[simp]
theorem mem_core (x : α) (s : Set β) : x ∈ f.core s ↔ ∀ y, y ∈ f x → y ∈ s :=
  Iff.rfl


theorem compl_dom_subset_core (s : Set β) : f.Domᶜ ⊆ f.core s := fun x hx y fxy =>
  absurd ((mem_dom f x).mpr ⟨y, fxy⟩) hx


theorem core_mono {s t : Set β} (h : s ⊆ t) : f.core s ⊆ f.core t :=
  Rel.core_mono _ h


theorem core_inter (s t : Set β) : f.core (s ∩ t) = f.core s ∩ f.core t :=
  Rel.core_inter _ s t


theorem mem_core_res (f : α → β) (s : Set α) (t : Set β) (x : α) :
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   f : α → β
                                                   s : Set α
                                                   t : Set β
                                                   x : α
                                                   ⊢ Iff (Membership.mem ((PFun.res f s).core t) x) (Membership.mem s x → Members …
                                                 -/
    x ∈ (res f s).core t ↔ x ∈ s → f x ∈ t := by simp [mem_core, mem_res]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem core_res (f : α → β) (s : Set α) (t : Set β) : (res f s).core t = sᶜ ∪ f ⁻¹' t := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    t : Set β
    ⊢ Eq ((PFun.res f s).core t) (Union.union (HasCompl.compl s) (Set.preimage f t))
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    t : Set β
    x : α
    ⊢ Iff (Membership.mem ((PFun.res f s).core t) x) (Membership.mem (Union.union  …
  -/
  rw [mem_core_res]
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set α
    t : Set β
    x : α
    ⊢ Iff (Membership.mem s x → Membership.mem t (f x)) (Membership.mem (Union.uni …
  -/
                         /-
                           🎉 no goals
                         -/
  by_cases h : x ∈ s <;> simp [h]
                         /-
                           🎉 no goals
                         -/


theorem core_restrict (f : α → β) (s : Set β) : (f : α →. β).core s = s.preimage f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    s : Set β
    ⊢ Eq ((↑f).core s) (Set.preimage f s)
  -/
  ext x; simp [core_def]
         /-
           🎉 no goals
         -/


theorem preimage_subset_core (f : α →. β) (s : Set β) : f.preimage s ⊆ f.core s :=
  fun _ ⟨y, ys, fxy⟩ y' fxy' =>
  have : y = y' := Part.mem_unique fxy fxy'
  this ▸ ys


theorem preimage_eq (f : α →. β) (s : Set β) : f.preimage s = f.core s ∩ f.Dom :=
  Set.eq_of_subset_of_subset (Set.subset_inter (f.preimage_subset_core s) (f.preimage_subset_dom s))
    fun x ⟨xcore, xdom⟩ =>
    let y := (f x).get xdom
    have ys : y ∈ s := xcore _ (Part.get_mem _)
    show x ∈ f.preimage s from ⟨(f x).get xdom, ys, Part.get_mem _⟩


theorem core_eq (f : α →. β) (s : Set β) : f.core s = f.preimage s ∪ f.Domᶜ := by
  rw [preimage_eq, Set.inter_union_distrib_right, Set.union_comm (Dom f), Set.compl_union_self,
    Set.inter_univ, Set.union_eq_self_of_subset_right (f.compl_dom_subset_core s)]


theorem preimage_asSubtype (f : α →. β) (s : Set β) :
    f.asSubtype ⁻¹' s = Subtype.val ⁻¹' f.preimage s := by
  /-
    α : Type u_1
    β : Type u_2
    f : PFun α β
    s : Set β
    ⊢ Eq (Set.preimage f.asSubtype s) (Set.preimage Subtype.val (f.preimage s))
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : PFun α β
    s : Set β
    x : ↑f.Dom
    ⊢ Iff (Membership.mem (Set.preimage f.asSubtype s) x) (Membership.mem (Set.pre …
  -/
  simp only [Set.mem_preimage, Set.mem_setOf_eq, PFun.asSubtype, PFun.mem_preimage]
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : PFun α β
    s : Set β
    x : ↑f.Dom
    ⊢ Iff (Membership.mem s (f.fn ↑x ⋯)) (Exists fun y => And (Membership.mem s y) …
  -/
  show f.fn x.val _ ∈ s ↔ ∃ y ∈ s, y ∈ f x.val
  exact
    Iff.intro (fun h => ⟨_, h, Part.get_mem _⟩) fun ⟨y, ys, fxy⟩ =>
      have : f.fn x.val x.property ∈ f x.val := Part.get_mem _
      Part.mem_unique fxy this ▸ ys


/-- Turns a function into a partial function to a subtype. -/
def toSubtype (p : β → Prop) (f : α → β) : α →. Subtype p := fun a => ⟨p (f a), Subtype.mk _⟩


@[simp]
theorem dom_toSubtype (p : β → Prop) (f : α → β) : (toSubtype p f).Dom = { a | p (f a) } :=
  rfl


@[simp]
theorem toSubtype_apply (p : β → Prop) (f : α → β) (a : α) :
    toSubtype p f a = ⟨p (f a), Subtype.mk _⟩ :=
  rfl


theorem dom_toSubtype_apply_iff {p : β → Prop} {f : α → β} {a : α} :
    (toSubtype p f a).Dom ↔ p (f a) :=
  Iff.rfl


theorem mem_toSubtype_iff {p : β → Prop} {f : α → β} {a : α} {b : Subtype p} :
    b ∈ toSubtype p f a ↔ ↑b = f a := by
  /-
    α : Type u_1
    β : Type u_2
    p : β → Prop
    f : α → β
    a : α
    b : Subtype p
    ⊢ Iff (Membership.mem (PFun.toSubtype p f a) b) (Eq (↑b) (f a))
  -/
  rw [toSubtype_apply, Part.mem_mk_iff, exists_subtype_mk_eq_iff, eq_comm]
  /-
    🎉 no goals
  -/


/-- The identity as a partial function -/
protected def id (α : Type*) : α →. α :=
  Part.some


@[simp]
theorem coe_id (α : Type*) : ((id : α → α) : α →. α) = PFun.id α :=
  rfl


@[simp]
theorem id_apply (a : α) : PFun.id α a = Part.some a :=
  rfl


/-- Composition of partial functions as a partial function. -/
def comp (f : β →. γ) (g : α →. β) : α →. γ := fun a => (g a).bind f


@[simp]
theorem comp_apply (f : β →. γ) (g : α →. β) (a : α) : f.comp g a = (g a).bind f :=
  rfl


@[simp]
theorem id_comp (f : α →. β) : (PFun.id β).comp f = f :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      f : PFun α β
                      x✝¹ : α
                      x✝ : β
                      ⊢ Iff (Membership.mem ((PFun.id β).comp f x✝¹) x✝) (Membership.mem (f x✝¹) x✝)
                    -/
  ext fun _ _ => by simp
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem comp_id (f : α →. β) : f.comp (PFun.id α) = f :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      f : PFun α β
                      x✝¹ : α
                      x✝ : β
                      ⊢ Iff (Membership.mem (f.comp (PFun.id α) x✝¹) x✝) (Membership.mem (f x✝¹) x✝)
                    -/
  ext fun _ _ => by simp
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem dom_comp (f : β →. γ) (g : α →. β) : (f.comp g).Dom = g.preimage f.Dom := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    ⊢ Eq (f.comp g).Dom (g.preimage f.Dom)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    x✝ : α
    ⊢ Iff (Membership.mem (f.comp g).Dom x✝) (Membership.mem (g.preimage f.Dom) x✝)
  -/
  simp_rw [mem_preimage, mem_dom, comp_apply, Part.mem_bind_iff, ← exists_and_right]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    x✝ : α
    ⊢ Iff (Exists fun y => Exists fun a => And (Membership.mem (g x✝) a) (Membersh …
  -/
  rw [exists_comm]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    x✝ : α
    ⊢ Iff (Exists fun b => Exists fun a => And (Membership.mem (g x✝) b) (Membersh …
  -/
  simp_rw [and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_comp (f : β →. γ) (g : α →. β) (s : Set γ) :
    (f.comp g).preimage s = g.preimage (f.preimage s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    s : Set γ
    ⊢ Eq ((f.comp g).preimage s) (g.preimage (f.preimage s))
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    s : Set γ
    x✝ : α
    ⊢ Iff (Membership.mem ((f.comp g).preimage s) x✝) (Membership.mem (g.preimage  …
  -/
  simp_rw [mem_preimage, comp_apply, Part.mem_bind_iff, ← exists_and_right, ← exists_and_left]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    s : Set γ
    x✝ : α
    ⊢ Iff (Exists fun y => Exists fun x => And (Membership.mem s y) (And (Membersh …
  -/
  rw [exists_comm]
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    s : Set γ
    x✝ : α
    ⊢ Iff (Exists fun b => Exists fun a => And (Membership.mem s a) (And (Membersh …
  -/
  simp_rw [and_assoc, and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem Part.bind_comp (f : β →. γ) (g : α →. β) (a : Part α) :
    a.bind (f.comp g) = (a.bind g).bind f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    a : Part α
    ⊢ Eq (a.bind (f.comp g)) ((a.bind g).bind f)
  -/
  ext c
  /-
    case H
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    a : Part α
    c : γ
    ⊢ Iff (Membership.mem (a.bind (f.comp g)) c) (Membership.mem ((a.bind g).bind  …
  -/
  simp_rw [Part.mem_bind_iff, comp_apply, Part.mem_bind_iff, ← exists_and_right, ← exists_and_left]
  /-
    case H
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    a : Part α
    c : γ
    ⊢ Iff (Exists fun a_1 => Exists fun x => And (Membership.mem a a_1) (And (Memb …
  -/
  rw [exists_comm]
  /-
    case H
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun β γ
    g : PFun α β
    a : Part α
    c : γ
    ⊢ Iff (Exists fun b => Exists fun a_1 => And (Membership.mem a a_1) (And (Memb …
  -/
  simp_rw [and_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_assoc (f : γ →. δ) (g : β →. γ) (h : α →. β) : (f.comp g).comp h = f.comp (g.comp h) :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      δ : Type u_4
                      f : PFun γ δ
                      g : PFun β γ
                      h : PFun α β
                      x✝¹ : α
                      x✝ : δ
                      ⊢ Iff (Membership.mem ((f.comp g).comp h x✝¹) x✝) (Membership.mem (f.comp (g.c …
                    -/
  ext fun _ _ => by simp only [comp_apply, Part.bind_comp]
                    /-
                      🎉 no goals
                    -/

-- This can't be `simp`

theorem coe_comp (g : β → γ) (f : α → β) : ((g ∘ f : α → γ) : α →. γ) = (g : β →. γ).comp f :=
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      g : β → γ
                      f : α → β
                      x✝¹ : α
                      x✝ : γ
                      ⊢ Iff (Membership.mem (↑(Function.comp g f) x✝¹) x✝) (Membership.mem ((↑g).com …
                    -/
  ext fun _ _ => by simp only [coe_val, comp_apply, Function.comp, Part.bind_some]
                    /-
                      🎉 no goals
                    -/


/-- Product of partial functions. -/
def prodLift (f : α →. β) (g : α →. γ) : α →. β × γ := fun x =>
  ⟨(f x).Dom ∧ (g x).Dom, fun h => ((f x).get h.1, (g x).get h.2)⟩


@[simp]
theorem dom_prodLift (f : α →. β) (g : α →. γ) :
    (f.prodLift g).Dom = { x | (f x).Dom ∧ (g x).Dom } :=
  rfl


theorem get_prodLift (f : α →. β) (g : α →. γ) (x : α) (h) :
    (f.prodLift g x).get h = ((f x).get h.1, (g x).get h.2) :=
  rfl


@[simp]
theorem prodLift_apply (f : α →. β) (g : α →. γ) (x : α) :
    f.prodLift g x = ⟨(f x).Dom ∧ (g x).Dom, fun h => ((f x).get h.1, (g x).get h.2)⟩ :=
  rfl


theorem mem_prodLift {f : α →. β} {g : α →. γ} {x : α} {y : β × γ} :
    y ∈ f.prodLift g x ↔ y.1 ∈ f x ∧ y.2 ∈ g x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : PFun α β
    g : PFun α γ
    x : α
    y : Prod β γ
    ⊢ Iff (Membership.mem (f.prodLift g x) y) (And (Membership.mem (f x) y.1) (Mem …
  -/
  trans ∃ hp hq, (f x).get hp = y.1 ∧ (g x).get hq = y.2
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : PFun α β
      g : PFun α γ
      x : α
      y : Prod β γ
      ⊢ Iff (Membership.mem (f.prodLift g x) y) (Exists fun hp => Exists fun hq => A …
    -/
  · simp only [prodLift, Part.mem_mk_iff, And.exists, Prod.ext_iff]
    /-
      🎉 no goals
    -/
  -- Porting note: was just `[exists_and_left, exists_and_right]`
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : PFun α β
      g : PFun α γ
      x : α
      y : Prod β γ
      ⊢ Iff (Exists fun hp => Exists fun hq => And (Eq ((f x).get hp) y.1) (Eq ((g x …
    -/
  · simp only [exists_and_left, exists_and_right, Membership.mem, Part.Mem]
    /-
      🎉 no goals
    -/


/-- Product of partial functions. -/
def prodMap (f : α →. γ) (g : β →. δ) : α × β →. γ × δ := fun x =>
  ⟨(f x.1).Dom ∧ (g x.2).Dom, fun h => ((f x.1).get h.1, (g x.2).get h.2)⟩


@[simp]
theorem dom_prodMap (f : α →. γ) (g : β →. δ) :
    (f.prodMap g).Dom = { x | (f x.1).Dom ∧ (g x.2).Dom } :=
  rfl


theorem get_prodMap (f : α →. γ) (g : β →. δ) (x : α × β) (h) :
    (f.prodMap g x).get h = ((f x.1).get h.1, (g x.2).get h.2) :=
  rfl


@[simp]
theorem prodMap_apply (f : α →. γ) (g : β →. δ) (x : α × β) :
    f.prodMap g x = ⟨(f x.1).Dom ∧ (g x.2).Dom, fun h => ((f x.1).get h.1, (g x.2).get h.2)⟩ :=
  rfl


theorem mem_prodMap {f : α →. γ} {g : β →. δ} {x : α × β} {y : γ × δ} :
    y ∈ f.prodMap g x ↔ y.1 ∈ f x.1 ∧ y.2 ∈ g x.2 := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    f : PFun α γ
    g : PFun β δ
    x : Prod α β
    y : Prod γ δ
    ⊢ Iff (Membership.mem (f.prodMap g x) y) (And (Membership.mem (f x.1) y.1) (Me …
  -/
  trans ∃ hp hq, (f x.1).get hp = y.1 ∧ (g x.2).get hq = y.2
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      f : PFun α γ
      g : PFun β δ
      x : Prod α β
      y : Prod γ δ
      ⊢ Iff (Membership.mem (f.prodMap g x) y) (Exists fun hp => Exists fun hq => An …
    -/
  · simp only [prodMap, Part.mem_mk_iff, And.exists, Prod.ext_iff]
    /-
      🎉 no goals
    -/
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      f : PFun α γ
      g : PFun β δ
      x : Prod α β
      y : Prod γ δ
      ⊢ Iff (Exists fun hp => Exists fun hq => And (Eq ((f x.1).get hp) y.1) (Eq ((g …
    -/
  · simp only [exists_and_left, exists_and_right, Membership.mem, Part.Mem]
    /-
      🎉 no goals
    -/


@[simp]
theorem prodLift_fst_comp_snd_comp (f : α →. γ) (g : β →. δ) :
    prodLift (f.comp ((Prod.fst : α × β → α) : α × β →. α))
        (g.comp ((Prod.snd : α × β → β) : α × β →. β)) =
      prodMap f g :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    δ : Type u_4
                    f : PFun α γ
                    g : PFun β δ
                    a : Prod α β
                    ⊢ ∀ (b : Prod γ δ), Iff (Membership.mem ((f.comp ↑Prod.fst).prodLift (g.comp ↑ …
                  -/
  ext fun a => by simp
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem prodMap_id_id : (PFun.id α).prodMap (PFun.id β) = PFun.id _ :=
                   /-
                     α : Type u_1
                     β : Type u_2
                     x✝¹ x✝ : Prod α β
                     ⊢ Iff (Membership.mem ((PFun.id α).prodMap (PFun.id β) x✝¹) x✝) (Membership.me …
                   -/
  ext fun _ _ ↦ by simp [eq_comm]
                   /-
                     🎉 no goals
                   -/


@[simp]
theorem prodMap_comp_comp (f₁ : α →. β) (f₂ : β →. γ) (g₁ : δ →. ε) (g₂ : ε →. ι) :
    (f₂.comp f₁).prodMap (g₂.comp g₁) = (f₂.prodMap g₂).comp (f₁.prodMap g₁) := -- by
  -- Porting note: was `by tidy`, below is a golfed version of the `tidy?` proof
  ext <| fun ⟨_, _⟩ ⟨_, _⟩ ↦
  ⟨fun ⟨⟨⟨h1l1, h1l2⟩, ⟨h1r1, h1r2⟩⟩, h2⟩ ↦ ⟨⟨⟨h1l1, h1r1⟩, ⟨h1l2, h1r2⟩⟩, h2⟩,
   fun ⟨⟨⟨h1l1, h1r1⟩, ⟨h1l2, h1r2⟩⟩, h2⟩ ↦ ⟨⟨⟨h1l1, h1l2⟩, ⟨h1r1, h1r2⟩⟩, h2⟩⟩


