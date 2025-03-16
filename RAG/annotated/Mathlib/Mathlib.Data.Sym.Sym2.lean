/-- This is the relation capturing the notion of pairs equivalent up to permutations. -/
@[aesop (rule_sets := [Sym2]) [safe [constructors, cases], norm]]
inductive Rel (α : Type u) : α × α → α × α → Prop
  | refl (x y : α) : Rel _ (x, y) (x, y)
  | swap (x y : α) : Rel _ (x, y) (y, x)


@[symm]
                                                             /-
                                                               α : Type u_1
                                                               x y : Prod α α
                                                               ⊢ Sym2.Rel α x y → Sym2.Rel α y x
                                                             -/
theorem Rel.symm {x y : α × α} : Rel α x y → Rel α y x := by aesop (rule_sets := [Sym2])
                                                             /-
                                                               🎉 no goals
                                                             -/


@[trans]
theorem Rel.trans {x y z : α × α} (a : Rel α x y) (b : Rel α y z) : Rel α x z := by
  /-
    α : Type u_1
    x y z : Prod α α
    a : Sym2.Rel α x y
    b : Sym2.Rel α y z
    ⊢ Sym2.Rel α x z
  -/
  aesop (rule_sets := [Sym2])
  /-
    🎉 no goals
  -/


theorem Rel.is_equivalence : Equivalence (Rel α) :=
  { refl := fun (x, y) ↦ Rel.refl x y, symm := Rel.symm, trans := Rel.trans }


/-- One can use `attribute [local instance] Sym2.Rel.setoid` to temporarily
make `Quotient` functionality work for `α × α`. -/
def Rel.setoid (α : Type u) : Setoid (α × α) :=
  ⟨Rel α, Rel.is_equivalence⟩


@[simp]
theorem rel_iff' {p q : α × α} : Rel α p q ↔ p = q ∨ p = q.swap := by
  /-
    α : Type u_1
    p q : Prod α α
    ⊢ Iff (Sym2.Rel α p q) (Or (Eq p q) (Eq p q.swap))
  -/
  aesop (rule_sets := [Sym2])
  /-
    🎉 no goals
  -/


theorem rel_iff {x y z w : α} : Rel α (x, y) (z, w) ↔ x = z ∧ y = w ∨ x = w ∧ y = z := by
  /-
    α : Type u_1
    x y z w : α
    ⊢ Iff (Sym2.Rel α { fst := x, snd := y } { fst := z, snd := w }) (Or (And (Eq  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- `Sym2 α` is the symmetric square of `α`, which, in other words, is the
type of unordered pairs.

It is equivalent in a natural way to multisets of cardinality 2 (see
`Sym2.equivMultiset`).
-/
abbrev Sym2 (α : Type u) := Quot (Sym2.Rel α)


/-- Constructor for `Sym2`. This is the quotient map `α × α → Sym2 α`. -/
protected abbrev Sym2.mk {α : Type*} (p : α × α) : Sym2 α := Quot.mk (Sym2.Rel α) p


/-- `s(x, y)` is an unordered pair,
which is to say a pair modulo the action of the symmetric group.

It is equal to `Sym2.mk (x, y)`. -/
notation3 "s(" x ", " y ")" => Sym2.mk (x, y)


protected theorem sound {p p' : α × α} (h : Sym2.Rel α p p') : Sym2.mk p = Sym2.mk p' :=
  Quot.sound h


protected theorem exact {p p' : α × α} (h : Sym2.mk p = Sym2.mk p') : Sym2.Rel α p p' :=
  Quotient.exact (s := Sym2.Rel.setoid α) h


@[simp]
protected theorem eq {p p' : α × α} : Sym2.mk p = Sym2.mk p' ↔ Sym2.Rel α p p' :=
  Quotient.eq' (s₁ := Sym2.Rel.setoid α)


@[elab_as_elim, cases_eliminator, induction_eliminator]
protected theorem ind {f : Sym2 α → Prop} (h : ∀ x y, f s(x, y)) : ∀ i, f i :=
  Quot.ind <| Prod.rec <| h


@[elab_as_elim]
protected theorem inductionOn {f : Sym2 α → Prop} (i : Sym2 α) (hf : ∀ x y, f s(x, y)) : f i :=
  i.ind hf


@[elab_as_elim]
protected theorem inductionOn₂ {f : Sym2 α → Sym2 β → Prop} (i : Sym2 α) (j : Sym2 β)
    (hf : ∀ a₁ a₂ b₁ b₂, f s(a₁, a₂) s(b₁, b₂)) : f i j :=
  Quot.induction_on₂ i j <| by
    /-
      α : Type u_1
      β : Type u_2
      f : Sym2 α → Sym2 β → Prop
      i : Sym2 α
      j : Sym2 β
      hf : ∀ (a₁ a₂ : α) (b₁ b₂ : β), f (Sym2.mk { fst := a₁, snd := a₂ }) (Sym2.mk  …
      ⊢ ∀ (a : Prod α α) (b : Prod β β), f (Quot.mk (Sym2.Rel α) a) (Quot.mk (Sym2.R …
    -/
    intro ⟨a₁, a₂⟩ ⟨b₁, b₂⟩
    /-
      α : Type u_1
      β : Type u_2
      f : Sym2 α → Sym2 β → Prop
      i : Sym2 α
      j : Sym2 β
      hf : ∀ (a₁ a₂ : α) (b₁ b₂ : β), f (Sym2.mk { fst := a₁, snd := a₂ }) (Sym2.mk  …
      a₁ a₂ : α
      b₁ b₂ : β
      ⊢ f (Quot.mk (Sym2.Rel α) { fst := a₁, snd := a₂ }) (Quot.mk (Sym2.Rel β) { fs …
    -/
    exact hf _ _ _ _
    /-
      🎉 no goals
    -/


/-- Dependent recursion principal for `Sym2`. See `Quot.rec`. -/
@[elab_as_elim]
protected def rec {motive : Sym2 α → Sort*}
    (f : (p : α × α) → motive (Sym2.mk p))
    (h : (p q : α × α) → (h : Sym2.Rel α p q) → Eq.ndrec (f p) (Sym2.sound h) = f q)
    (z : Sym2 α) : motive z :=
  Quot.rec f h z


/-- Dependent recursion principal for `Sym2` when the target is a `Subsingleton` type.
See `Quot.recOnSubsingleton`. -/
@[elab_as_elim]
protected abbrev recOnSubsingleton {motive : Sym2 α → Sort*}
    [(p : α × α) → Subsingleton (motive (Sym2.mk p))]
    (z : Sym2 α) (f : (p : α × α) → motive (Sym2.mk p)) : motive z :=
  Quot.recOnSubsingleton z f


protected theorem «exists» {α : Sort _} {f : Sym2 α → Prop} :
    (∃ x : Sym2 α, f x) ↔ ∃ x y, f s(x, y) :=
  Quot.mk_surjective.exists.trans Prod.exists


protected theorem «forall» {α : Sort _} {f : Sym2 α → Prop} :
    (∀ x : Sym2 α, f x) ↔ ∀ x y, f s(x, y) :=
  Quot.mk_surjective.forall.trans Prod.forall


theorem eq_swap {a b : α} : s(a, b) = s(b, a) := Quot.sound (Rel.swap _ _)


@[simp]
theorem mk_prod_swap_eq {p : α × α} : Sym2.mk p.swap = Sym2.mk p := by
  /-
    α : Type u_1
    p : Prod α α
    ⊢ Eq (Sym2.mk p.swap) (Sym2.mk p)
  -/
  cases p
  /-
    case mk
    α : Type u_1
    fst✝ snd✝ : α
    ⊢ Eq (Sym2.mk { fst := fst✝, snd := snd✝ }.swap) (Sym2.mk { fst := fst✝, snd : …
  -/
  exact eq_swap
  /-
    🎉 no goals
  -/


theorem congr_right {a b c : α} : s(a, b) = s(a, c) ↔ b = c := by
  /-
    α : Type u_1
    a b c : α
    ⊢ Iff (Eq (Sym2.mk { fst := a, snd := b }) (Sym2.mk { fst := a, snd := c })) ( …
  -/
  simp (config := {contextual := true})
  /-
    🎉 no goals
  -/


theorem congr_left {a b c : α} : s(b, a) = s(c, a) ↔ b = c := by
  /-
    α : Type u_1
    a b c : α
    ⊢ Iff (Eq (Sym2.mk { fst := b, snd := a }) (Sym2.mk { fst := c, snd := a })) ( …
  -/
  simp (config := {contextual := true})
  /-
    🎉 no goals
  -/


theorem eq_iff {x y z w : α} : s(x, y) = s(z, w) ↔ x = z ∧ y = w ∨ x = w ∧ y = z := by
  /-
    α : Type u_1
    x y z w : α
    ⊢ Iff (Eq (Sym2.mk { fst := x, snd := y }) (Sym2.mk { fst := z, snd := w })) ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem mk_eq_mk_iff {p q : α × α} : Sym2.mk p = Sym2.mk q ↔ p = q ∨ p = q.swap := by
  /-
    α : Type u_1
    p q : Prod α α
    ⊢ Iff (Eq (Sym2.mk p) (Sym2.mk q)) (Or (Eq p q) (Eq p q.swap))
  -/
  cases p
  /-
    case mk
    α : Type u_1
    q : Prod α α
    fst✝ snd✝ : α
    ⊢ Iff (Eq (Sym2.mk { fst := fst✝, snd := snd✝ }) (Sym2.mk q)) (Or (Eq { fst := …
  -/
  cases q
  /-
    case mk.mk
    α : Type u_1
    fst✝¹ snd✝¹ fst✝ snd✝ : α
    ⊢ Iff (Eq (Sym2.mk { fst := fst✝¹, snd := snd✝¹ }) (Sym2.mk { fst := fst✝, snd …
  -/
  simp only [eq_iff, Prod.mk.inj_iff, Prod.swap_prod_mk]
  /-
    🎉 no goals
  -/


/-- The universal property of `Sym2`; symmetric functions of two arguments are equivalent to
functions from `Sym2`. Note that when `β` is `Prop`, it can sometimes be more convenient to use
`Sym2.fromRel` instead. -/
def lift : { f : α → α → β // ∀ a₁ a₂, f a₁ a₂ = f a₂ a₁ } ≃ (Sym2 α → β) where
  toFun f :=
    Quot.lift (uncurry ↑f) <| by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : Subtype fun f => ∀ (a₁ a₂ : α), Eq (f a₁ a₂) (f a₂ a₁)
        ⊢ ∀ (a b : Prod α α), Sym2.Rel α a b → Eq (Function.uncurry (↑f) a) (Function. …
      -/
      rintro _ _ ⟨⟩
      /-
        case refl
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        f : Subtype fun f => ∀ (a₁ a₂ : α), Eq (f a₁ a₂) (f a₂ a₁)
        x✝ y✝ : α
        ⊢ Eq (Function.uncurry ↑f { fst := x✝, snd := y✝ }) (Function.uncurry ↑f { fst …
      -/
      exacts [rfl, f.prop _ _]
      /-
        🎉 no goals
      -/
  invFun F := ⟨curry (F ∘ Sym2.mk), fun _ _ => congr_arg F eq_swap⟩
  left_inv _ := Subtype.ext rfl
  right_inv _ := funext <| Sym2.ind fun _ _ => rfl


@[simp]
theorem lift_mk (f : { f : α → α → β // ∀ a₁ a₂, f a₁ a₂ = f a₂ a₁ }) (a₁ a₂ : α) :
    lift f s(a₁, a₂) = (f : α → α → β) a₁ a₂ :=
  rfl


@[simp]
theorem coe_lift_symm_apply (F : Sym2 α → β) (a₁ a₂ : α) :
    (lift.symm F : α → α → β) a₁ a₂ = F s(a₁, a₂) :=
  rfl


/-- A two-argument version of `Sym2.lift`. -/
def lift₂ :
    { f : α → α → β → β → γ //
        ∀ a₁ a₂ b₁ b₂, f a₁ a₂ b₁ b₂ = f a₂ a₁ b₁ b₂ ∧ f a₁ a₂ b₁ b₂ = f a₁ a₂ b₂ b₁ } ≃
      (Sym2 α → Sym2 β → γ) where
  toFun f :=
    Quotient.lift₂ (s₁ := Sym2.Rel.setoid α) (s₂ := Sym2.Rel.setoid β)
      (fun (a : α × α) (b : β × β) => f.1 a.1 a.2 b.1 b.2)
      (by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          f : Subtype fun f => ∀ (a₁ a₂ : α) (b₁ b₂ : β), And (Eq (f a₁ a₂ b₁ b₂) (f a₂  …
          ⊢ ∀ (a₁ : Prod α α) (b₁ : Prod β β) (a₂ : Prod α α) (b₂ : Prod β β), HasEquiv. …
        -/
        rintro _ _ _ _ ⟨⟩ ⟨⟩
        /-
          case refl.refl
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          f : Subtype fun f => ∀ (a₁ a₂ : α) (b₁ b₂ : β), And (Eq (f a₁ a₂ b₁ b₂) (f a₂  …
          x✝¹ y✝¹ : α
          x✝ y✝ : β
          ⊢ Eq ((fun a b => ↑f a.1 a.2 b.1 b.2) { fst := x✝¹, snd := y✝¹ } { fst := x✝,  …
        -/
        exacts [rfl, (f.2 _ _ _ _).2, (f.2 _ _ _ _).1, (f.2 _ _ _ _).1.trans (f.2 _ _ _ _).2])
        /-
          🎉 no goals
        -/
  invFun F :=
    ⟨fun a₁ a₂ b₁ b₂ => F s(a₁, a₂) s(b₁, b₂), fun a₁ a₂ b₁ b₂ => by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        F : Sym2 α → Sym2 β → γ
        a₁ a₂ : α
        b₁ b₂ : β
        ⊢ And (Eq ((fun a₁ a₂ b₁ b₂ => F (Sym2.mk { fst := a₁, snd := a₂ }) (Sym2.mk { …
      -/
      constructor
      /-
        case left
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        F : Sym2 α → Sym2 β → γ
        a₁ a₂ : α
        b₁ b₂ : β
        ⊢ Eq ((fun a₁ a₂ b₁ b₂ => F (Sym2.mk { fst := a₁, snd := a₂ }) (Sym2.mk { fst  …
      -/
      exacts [congr_arg₂ F eq_swap rfl, congr_arg₂ F rfl eq_swap]⟩
      /-
        🎉 no goals
      -/
  left_inv _ := Subtype.ext rfl
  right_inv _ := funext₂ fun a b => Sym2.inductionOn₂ a b fun _ _ _ _ => rfl


@[simp]
theorem lift₂_mk
    (f :
    { f : α → α → β → β → γ //
      ∀ a₁ a₂ b₁ b₂, f a₁ a₂ b₁ b₂ = f a₂ a₁ b₁ b₂ ∧ f a₁ a₂ b₁ b₂ = f a₁ a₂ b₂ b₁ })
    (a₁ a₂ : α) (b₁ b₂ : β) : lift₂ f s(a₁, a₂) s(b₁, b₂) = (f : α → α → β → β → γ) a₁ a₂ b₁ b₂ :=
  rfl


@[simp]
theorem coe_lift₂_symm_apply (F : Sym2 α → Sym2 β → γ) (a₁ a₂ : α) (b₁ b₂ : β) :
    (lift₂.symm F : α → α → β → β → γ) a₁ a₂ b₁ b₂ = F s(a₁, a₂) s(b₁, b₂) :=
  rfl


/-- The functor `Sym2` is functorial, and this function constructs the induced maps.
-/
def map (f : α → β) : Sym2 α → Sym2 β :=
  Quot.map (Prod.map f f)
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          f : α → β
          ⊢ ∀ ⦃a b : Prod α α⦄, Sym2.Rel α a b → Sym2.Rel β (Prod.map f f a) (Prod.map f …
        -/
                                 /-
                                   🎉 no goals
                                 -/
    (by intro _ _ h; cases h <;> constructor)
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem map_id : map (@id α) = id := by
  /-
    α : Type u_1
    ⊢ Eq (Sym2.map id) id
  -/
  ext ⟨⟨x, y⟩⟩
  /-
    case h.mk.mk
    α : Type u_1
    x✝ : Sym2 α
    x y : α
    ⊢ Eq (Sym2.map id (Quot.mk (Sym2.Rel α) { fst := x, snd := y })) (id (Quot.mk  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map_comp {g : β → γ} {f : α → β} : Sym2.map (g ∘ f) = Sym2.map g ∘ Sym2.map f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    g : β → γ
    f : α → β
    ⊢ Eq (Sym2.map (Function.comp g f)) (Function.comp (Sym2.map g) (Sym2.map f))
  -/
  ext ⟨⟨x, y⟩⟩
  /-
    case h.mk.mk
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    g : β → γ
    f : α → β
    x✝ : Sym2 α
    x y : α
    ⊢ Eq (Sym2.map (Function.comp g f) (Quot.mk (Sym2.Rel α) { fst := x, snd := y  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem map_map {g : β → γ} {f : α → β} (x : Sym2 α) : map g (map f x) = map (g ∘ f) x := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    g : β → γ
    f : α → β
    x : Sym2 α
    ⊢ Eq (Sym2.map g (Sym2.map f x)) (Sym2.map (Function.comp g f) x)
  -/
  induction x; aesop
               /-
                 🎉 no goals
               -/


@[simp]
theorem map_pair_eq (f : α → β) (x y : α) : map f s(x, y) = s(f x, f y) :=
  rfl


theorem map.injective {f : α → β} (hinj : Injective f) : Injective (map f) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hinj : Function.Injective f
    ⊢ Function.Injective (Sym2.map f)
  -/
  intro z z'
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hinj : Function.Injective f
    z z' : Sym2 α
    ⊢ Eq (Sym2.map f z) (Sym2.map f z') → Eq z z'
  -/
  refine Sym2.inductionOn₂ z z' (fun x y x' y' => ?_)
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hinj : Function.Injective f
    z z' : Sym2 α
    x y x' y' : α
    ⊢ Eq (Sym2.map f (Sym2.mk { fst := x, snd := y })) (Sym2.map f (Sym2.mk { fst  …
  -/
  simp [hinj.eq_iff]
  /-
    🎉 no goals
  -/


/-- `mk a` as an embedding. This is the symmetric version of `Function.Embedding.sectL`. -/
@[simps]
def mkEmbedding (a : α) : α ↪ Sym2 α where
  toFun b := s(a, b)
  inj' b₁ b₁ h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      a b₁✝ b₁ : α
      h : Eq ((fun b => Sym2.mk { fst := a, snd := b }) b₁✝) ((fun b => Sym2.mk { fs …
      ⊢ Eq b₁✝ b₁
    -/
    simp only [Sym2.eq, Sym2.rel_iff', Prod.mk.injEq, true_and, Prod.swap_prod_mk] at h
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      a b₁✝ b₁ : α
      h : Or (Eq b₁✝ b₁) (And (Eq a b₁) (Eq b₁✝ a))
      ⊢ Eq b₁✝ b₁
    -/
                                     /-
                                       🎉 no goals
                                     -/
    obtain rfl | ⟨rfl, rfl⟩ := h <;> rfl
                                     /-
                                       🎉 no goals
                                     -/


/-- `Sym2.map` as an embedding. -/
@[simps]
def _root_.Function.Embedding.sym2Map (f : α ↪ β) : Sym2 α ↪ Sym2 β where
  toFun := map f
  inj' := map.injective f.injective


/-- This is a predicate that determines whether a given term is a member of a term of the
symmetric square.  From this point of view, the symmetric square is the subtype of
cardinality-two multisets on `α`.
-/
protected def Mem (x : α) (z : Sym2 α) : Prop :=
  ∃ y : α, z = s(x, y)


@[aesop norm (rule_sets := [Sym2])]
theorem mem_iff' {a b c : α} : Sym2.Mem a s(b, c) ↔ a = b ∨ a = c :=
  { mp := by
      /-
        α : Type u_1
        a b c : α
        ⊢ Sym2.Mem a (Sym2.mk { fst := b, snd := c }) → Or (Eq a b) (Eq a c)
      -/
      rintro ⟨_, h⟩
      /-
        case intro
        α : Type u_1
        a b c w✝ : α
        h : Eq (Sym2.mk { fst := b, snd := c }) (Sym2.mk { fst := a, snd := w✝ })
        ⊢ Or (Eq a b) (Eq a c)
      -/
      rw [eq_iff] at h
      /-
        case intro
        α : Type u_1
        a b c w✝ : α
        h : Or (And (Eq b a) (Eq c w✝)) (And (Eq b w✝) (Eq c a))
        ⊢ Or (Eq a b) (Eq a c)
      -/
      aesop
      /-
        🎉 no goals
      -/
    mpr := by
      /-
        α : Type u_1
        a b c : α
        ⊢ Or (Eq a b) (Eq a c) → Sym2.Mem a (Sym2.mk { fst := b, snd := c })
      -/
      rintro (rfl | rfl)
        /-
          case inl
          α : Type u_1
          a c : α
          ⊢ Sym2.Mem a (Sym2.mk { fst := a, snd := c })
        -/
      · exact ⟨_, rfl⟩
        /-
          🎉 no goals
        -/
      /-
        case inr
        α : Type u_1
        a b : α
        ⊢ Sym2.Mem a (Sym2.mk { fst := b, snd := a })
      -/
      rw [eq_swap]
      /-
        case inr
        α : Type u_1
        a b : α
        ⊢ Sym2.Mem a (Sym2.mk { fst := a, snd := b })
      -/
      exact ⟨_, rfl⟩ }
      /-
        🎉 no goals
      -/


instance : SetLike (Sym2 α) α where
  coe z := { x | z.Mem x }
  coe_injective' z z' h := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      z z' : Sym2 α
      h : Eq ((fun z => setOf fun x => Sym2.Mem x z) z) ((fun z => setOf fun x => Sy …
      ⊢ Eq z z'
    -/
    simp only [Set.ext_iff, Set.mem_setOf_eq] at h
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      z z' : Sym2 α
      h : ∀ (x : α), Iff (Sym2.Mem x z) (Sym2.Mem x z')
      ⊢ Eq z z'
    -/
    induction' z with x y
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      z' : Sym2 α
      x y : α
      h : ∀ (x_1 : α), Iff (Sym2.Mem x_1 (Sym2.mk { fst := x, snd := y })) (Sym2.Mem …
      ⊢ Eq (Sym2.mk { fst := x, snd := y }) z'
    -/
    induction' z' with x' y'
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      x y x' y' : α
      h : ∀ (x_1 : α), Iff (Sym2.Mem x_1 (Sym2.mk { fst := x, snd := y })) (Sym2.Mem …
      ⊢ Eq (Sym2.mk { fst := x, snd := y }) (Sym2.mk { fst := x', snd := y' })
    -/
    have hx := h x; have hy := h y; have hx' := h x'; have hy' := h y'
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      x y x' y' : α
      h : ∀ (x_1 : α), Iff (Sym2.Mem x_1 (Sym2.mk { fst := x, snd := y })) (Sym2.Mem …
      hx : Iff (Sym2.Mem x (Sym2.mk { fst := x, snd := y })) (Sym2.Mem x (Sym2.mk {  …
      hy : Iff (Sym2.Mem y (Sym2.mk { fst := x, snd := y })) (Sym2.Mem y (Sym2.mk {  …
      hx' : Iff (Sym2.Mem x' (Sym2.mk { fst := x, snd := y })) (Sym2.Mem x' (Sym2.mk …
      hy' : Iff (Sym2.Mem y' (Sym2.mk { fst := x, snd := y })) (Sym2.Mem y' (Sym2.mk …
      ⊢ Eq (Sym2.mk { fst := x, snd := y }) (Sym2.mk { fst := x', snd := y' })
    -/
    simp only [mem_iff', eq_self_iff_true] at hx hy hx' hy'
    /-
      case h.h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      x y x' y' : α
      h : ∀ (x_1 : α), Iff (Sym2.Mem x_1 (Sym2.mk { fst := x, snd := y })) (Sym2.Mem …
      hx : Iff (Or True (Eq x y)) (Or (Eq x x') (Eq x y'))
      hy : Iff (Or (Eq y x) True) (Or (Eq y x') (Eq y y'))
      hx' : Iff (Or (Eq x' x) (Eq x' y)) (Or True (Eq x' y'))
      hy' : Iff (Or (Eq y' x) (Eq y' y)) (Or (Eq y' x') True)
      ⊢ Eq (Sym2.mk { fst := x, snd := y }) (Sym2.mk { fst := x', snd := y' })
    -/
    aesop
    /-
      🎉 no goals
    -/


@[simp]
theorem mem_iff_mem {x : α} {z : Sym2 α} : Sym2.Mem x z ↔ x ∈ z :=
  Iff.rfl


theorem mem_iff_exists {x : α} {z : Sym2 α} : x ∈ z ↔ ∃ y : α, z = s(x, y) :=
  Iff.rfl


@[ext]
theorem ext {p q : Sym2 α} (h : ∀ x, x ∈ p ↔ x ∈ q) : p = q :=
  SetLike.ext h


theorem mem_mk_left (x y : α) : x ∈ s(x, y) :=
  ⟨y, rfl⟩


theorem mem_mk_right (x y : α) : y ∈ s(x, y) :=
  eq_swap ▸ mem_mk_left y x


@[simp, aesop norm (rule_sets := [Sym2])]
theorem mem_iff {a b c : α} : a ∈ s(b, c) ↔ a = b ∨ a = c :=
  mem_iff'


theorem out_fst_mem (e : Sym2 α) : e.out.1 ∈ e :=
               /-
                 α : Type u_1
                 e : Sym2 α
                 ⊢ Eq e (Sym2.mk { fst := (Quot.out e).1, snd := (Quot.out e).2 })
               -/
  ⟨e.out.2, by rw [Sym2.mk, e.out_eq]⟩
               /-
                 🎉 no goals
               -/


theorem out_snd_mem (e : Sym2 α) : e.out.2 ∈ e :=
               /-
                 α : Type u_1
                 e : Sym2 α
                 ⊢ Eq e (Sym2.mk { fst := (Quot.out e).2, snd := (Quot.out e).1 })
               -/
  ⟨e.out.1, by rw [eq_swap, Sym2.mk, e.out_eq]⟩
               /-
                 🎉 no goals
               -/


theorem ball {p : α → Prop} {a b : α} : (∀ c ∈ s(a, b), p c) ↔ p a ∧ p b := by
  /-
    α : Type u_1
    p : α → Prop
    a b : α
    ⊢ Iff (∀ (c : α), Membership.mem (Sym2.mk { fst := a, snd := b }) c → p c) (An …
  -/
  refine ⟨fun h => ⟨h _ <| mem_mk_left _ _, h _ <| mem_mk_right _ _⟩, fun h c hc => ?_⟩
  /-
    α : Type u_1
    p : α → Prop
    a b : α
    h : And (p a) (p b)
    c : α
    hc : Membership.mem (Sym2.mk { fst := a, snd := b }) c
    ⊢ p c
  -/
  obtain rfl | rfl := Sym2.mem_iff.1 hc
    /-
      case inl
      α : Type u_1
      p : α → Prop
      b c : α
      h : And (p c) (p b)
      hc : Membership.mem (Sym2.mk { fst := c, snd := b }) c
      ⊢ p c
    -/
  · exact h.1
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      p : α → Prop
      a c : α
      h : And (p a) (p c)
      hc : Membership.mem (Sym2.mk { fst := a, snd := c }) c
      ⊢ p c
    -/
  · exact h.2
    /-
      🎉 no goals
    -/


/-- Given an element of the unordered pair, give the other element using `Classical.choose`.
See also `Mem.other'` for the computable version.
-/
noncomputable def Mem.other {a : α} {z : Sym2 α} (h : a ∈ z) : α :=
  Classical.choose h


@[simp]
theorem other_spec {a : α} {z : Sym2 α} (h : a ∈ z) : s(a, Mem.other h) = z := by
  /-
    α : Type u_1
    a : α
    z : Sym2 α
    h : Membership.mem z a
    ⊢ Eq (Sym2.mk { fst := a, snd := Sym2.Mem.other h }) z
  -/
  erw [← Classical.choose_spec h]
  /-
    🎉 no goals
  -/


theorem other_mem {a : α} {z : Sym2 α} (h : a ∈ z) : Mem.other h ∈ z := by
  /-
    α : Type u_1
    a : α
    z : Sym2 α
    h : Membership.mem z a
    ⊢ Membership.mem z (Sym2.Mem.other h)
  -/
  convert mem_mk_right a <| Mem.other h
  /-
    case h.e'_4
    α : Type u_1
    a : α
    z : Sym2 α
    h : Membership.mem z a
    ⊢ Eq z (Sym2.mk { fst := a, snd := Sym2.Mem.other h })
  -/
  rw [other_spec h]
  /-
    🎉 no goals
  -/


theorem mem_and_mem_iff {x y : α} {z : Sym2 α} (hne : x ≠ y) : x ∈ z ∧ y ∈ z ↔ z = s(x, y) := by
  /-
    α : Type u_1
    x y : α
    z : Sym2 α
    hne : Ne x y
    ⊢ Iff (And (Membership.mem z x) (Membership.mem z y)) (Eq z (Sym2.mk { fst :=  …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      x y : α
      z : Sym2 α
      hne : Ne x y
      ⊢ And (Membership.mem z x) (Membership.mem z y) → Eq z (Sym2.mk { fst := x, sn …
    -/
  · induction' z with x' y'
    /-
      case mp.h
      α : Type u_1
      x y : α
      hne : Ne x y
      x' y' : α
      ⊢ And (Membership.mem (Sym2.mk { fst := x', snd := y' }) x) (Membership.mem (S …
    -/
    rw [mem_iff, mem_iff]
    /-
      case mp.h
      α : Type u_1
      x y : α
      hne : Ne x y
      x' y' : α
      ⊢ And (Or (Eq x x') (Eq x y')) (Or (Eq y x') (Eq y y')) → Eq (Sym2.mk { fst := …
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      x y : α
      z : Sym2 α
      hne : Ne x y
      ⊢ Eq z (Sym2.mk { fst := x, snd := y }) → And (Membership.mem z x) (Membership …
    -/
  · rintro rfl
    /-
      case mpr
      α : Type u_1
      x y : α
      hne : Ne x y
      ⊢ And (Membership.mem (Sym2.mk { fst := x, snd := y }) x) (Membership.mem (Sym …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem eq_of_ne_mem {x y : α} {z z' : Sym2 α} (h : x ≠ y) (h1 : x ∈ z) (h2 : y ∈ z) (h3 : x ∈ z')
    (h4 : y ∈ z') : z = z' :=
  ((mem_and_mem_iff h).mp ⟨h1, h2⟩).trans ((mem_and_mem_iff h).mp ⟨h3, h4⟩).symm


instance Mem.decidable [DecidableEq α] (x : α) (z : Sym2 α) : Decidable (x ∈ z) :=
  z.recOnSubsingleton fun ⟨_, _⟩ => decidable_of_iff' _ mem_iff


@[simp]
theorem mem_map {f : α → β} {b : β} {z : Sym2 α} : b ∈ Sym2.map f z ↔ ∃ a, a ∈ z ∧ f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    z : Sym2 α
    ⊢ Iff (Membership.mem (Sym2.map f z) b) (Exists fun a => And (Membership.mem z …
  -/
  induction' z with x y
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    x y : α
    ⊢ Iff (Membership.mem (Sym2.map f (Sym2.mk { fst := x, snd := y })) b) (Exists …
  -/
  simp only [map_pair_eq, mem_iff, exists_eq_or_imp, exists_eq_left]
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    b : β
    x y : α
    ⊢ Iff (Or (Eq b (f x)) (Eq b (f y))) (Or (Eq (f x) b) (Eq (f y) b))
  -/
  aesop
  /-
    🎉 no goals
  -/


@[congr]
theorem map_congr {f g : α → β} {s : Sym2 α} (h : ∀ x ∈ s, f x = g x) : map f s = map g s := by
  /-
    α : Type u_1
    β : Type u_2
    f g : α → β
    s : Sym2 α
    h : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
    ⊢ Eq (Sym2.map f s) (Sym2.map g s)
  -/
  ext y
  /-
    case h
    α : Type u_1
    β : Type u_2
    f g : α → β
    s : Sym2 α
    h : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
    y : β
    ⊢ Iff (Membership.mem (Sym2.map f s) y) (Membership.mem (Sym2.map g s) y)
  -/
  simp only [mem_map]
  /-
    case h
    α : Type u_1
    β : Type u_2
    f g : α → β
    s : Sym2 α
    h : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
    y : β
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (Eq (f a) y)) (Exists fun a => …
  -/
  constructor <;>
      /-
        case h.mp
        α : Type u_1
        β : Type u_2
        f g : α → β
        s : Sym2 α
        h : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
        y : β
        ⊢ (Exists fun a => And (Membership.mem s a) (Eq (f a) y)) → Exists fun a => An …
      -/
      /-
        case h.mp.intro.intro
        α : Type u_1
        β : Type u_2
        f g : α → β
        s : Sym2 α
        h : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
        w : α
        hw : Membership.mem s w
        ⊢ Exists fun a => And (Membership.mem s a) (Eq (g a) (f w))
      -/
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.intro.intro
        α : Type u_1
        β : Type u_2
        f g : α → β
        s : Sym2 α
        h : ∀ (x : α), Membership.mem s x → Eq (f x) (g x)
        w : α
        hw : Membership.mem s w
        ⊢ Exists fun a => And (Membership.mem s a) (Eq (f a) (g w))
      -/
      exact ⟨w, hw, by simp [hw, h]⟩
      /-
        🎉 no goals
      -/


/-- Note: `Sym2.map_id` will not simplify `Sym2.map id z` due to `Sym2.map_congr`. -/
@[simp]
theorem map_id' : (map fun x : α => x) = id :=
  map_id


/--
Partial map. If `f : ∀ a, p a → β` is a partial function defined on `a : α` satisfying `p`,
then `pmap f s h` is essentially the same as `map f s` but is defined only when all members of `s`
satisfy `p`, using the proof to apply `f`.
-/
def pmap {P : α → Prop} (f : ∀ a, P a → β) (s : Sym2 α) : (∀ a ∈ s, P a) → Sym2 β :=
  let g (p : α × α) (H : ∀ a ∈ Sym2.mk p, P a) : Sym2 β :=
    s(f p.1 (H p.1 <| mem_mk_left _ _), f p.2 (H p.2 <| mem_mk_right _ _))
  Quot.recOn s g fun p q hpq => funext fun Hq => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      P : α → Prop
      f : (a : α) → P a → β
      s : Sym2 α
      g : (p : Prod α α) → (∀ (a : α), Membership.mem (Sym2.mk p) a → P a) → Sym2 β  …
      p q : Prod α α
      hpq : Sym2.Rel α p q
      Hq : ∀ (a : α), Membership.mem (Quot.mk (Sym2.Rel α) q) a → P a
      ⊢ Eq (Eq.ndrec (motive := fun x => (∀ (a : α), Membership.mem x a → P a) → Sym …
    -/
    rw [rel_iff'] at hpq
    have Hp : ∀ a ∈ Sym2.mk p, P a := fun a hmem =>
      Hq a (Sym2.mk_eq_mk_iff.2 hpq ▸ hmem : a ∈ Sym2.mk q)
    have h : ∀ {s₂ e H}, Eq.ndrec (motive := fun s => (∀ a ∈ s, P a) → Sym2 β) (g p) (b := s₂) e H =
      g p Hp := by
      rintro s₂ rfl _
      rfl
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      P : α → Prop
      f : (a : α) → P a → β
      s : Sym2 α
      g : (p : Prod α α) → (∀ (a : α), Membership.mem (Sym2.mk p) a → P a) → Sym2 β  …
      p q : Prod α α
      hpq✝ : Sym2.Rel α p q
      hpq : Or (Eq p q) (Eq p q.swap)
      Hq : ∀ (a : α), Membership.mem (Quot.mk (Sym2.Rel α) q) a → P a
      Hp : ∀ (a : α), Membership.mem (Sym2.mk p) a → P a
      h : ∀ {s₂ : Sym2 α} {e : Eq (Sym2.mk p) s₂} {H : ∀ (a : α), Membership.mem s₂  …
      ⊢ Eq (Eq.ndrec (motive := fun x => (∀ (a : α), Membership.mem x a → P a) → Sym …
    -/
    refine h.trans (Quot.sound ?_)
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      P : α → Prop
      f : (a : α) → P a → β
      s : Sym2 α
      g : (p : Prod α α) → (∀ (a : α), Membership.mem (Sym2.mk p) a → P a) → Sym2 β  …
      p q : Prod α α
      hpq✝ : Sym2.Rel α p q
      hpq : Or (Eq p q) (Eq p q.swap)
      Hq : ∀ (a : α), Membership.mem (Quot.mk (Sym2.Rel α) q) a → P a
      Hp : ∀ (a : α), Membership.mem (Sym2.mk p) a → P a
      h : ∀ {s₂ : Sym2 α} {e : Eq (Sym2.mk p) s₂} {H : ∀ (a : α), Membership.mem s₂  …
      ⊢ Sym2.Rel β { fst := f p.1 ⋯, snd := f p.2 ⋯ } { fst := f q.1 ⋯, snd := f q.2 …
    -/
    rw [rel_iff', Prod.mk.injEq, Prod.swap_prod_mk]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      P : α → Prop
      f : (a : α) → P a → β
      s : Sym2 α
      g : (p : Prod α α) → (∀ (a : α), Membership.mem (Sym2.mk p) a → P a) → Sym2 β  …
      p q : Prod α α
      hpq✝ : Sym2.Rel α p q
      hpq : Or (Eq p q) (Eq p q.swap)
      Hq : ∀ (a : α), Membership.mem (Quot.mk (Sym2.Rel α) q) a → P a
      Hp : ∀ (a : α), Membership.mem (Sym2.mk p) a → P a
      h : ∀ {s₂ : Sym2 α} {e : Eq (Sym2.mk p) s₂} {H : ∀ (a : α), Membership.mem s₂  …
      ⊢ Or (And (Eq (f p.1 ⋯) (f q.1 ⋯)) (Eq (f p.2 ⋯) (f q.2 ⋯))) (Eq { fst := f p. …
    -/
                                     /-
                                       🎉 no goals
                                     -/
    apply hpq.imp <;> rintro rfl <;> simp
                                     /-
                                       🎉 no goals
                                     -/


theorem forall_mem_pair {P : α → Prop} {a b : α} : (∀ x ∈ s(a, b), P x) ↔ P a ∧ P b := by
  /-
    α : Type u_1
    P : α → Prop
    a b : α
    ⊢ Iff (∀ (x : α), Membership.mem (Sym2.mk { fst := a, snd := b }) x → P x) (An …
  -/
  simp only [mem_iff, forall_eq_or_imp, forall_eq]
  /-
    🎉 no goals
  -/


lemma pair_eq_pmap {P : α → Prop} (f : ∀ a, P a → β) (a b : α) (h : P a) (h' : P b) :
    s(f a h, f b h') = pmap f s(a, b) (forall_mem_pair.mpr ⟨h, h'⟩) := rfl


lemma pmap_pair {P : α → Prop} (f : ∀ a, P a → β) (a b : α) (h : ∀ x ∈ s(a, b), P x) :
    pmap f s(a, b) h = s(f a (h a (mem_mk_left a b)), f b (h b (mem_mk_right a b))) := rfl


@[simp]
lemma mem_pmap_iff {P : α → Prop} (f : ∀ a, P a → β) (z : Sym2 α) (h : ∀ a ∈ z, P a) (b : β) :
    b ∈ z.pmap f h ↔ ∃ (a : α) (ha : a ∈ z), b = f a (h a ha) := by
  /-
    α : Type u_1
    β : Type u_2
    P : α → Prop
    f : (a : α) → P a → β
    z : Sym2 α
    h : ∀ (a : α), Membership.mem z a → P a
    b : β
    ⊢ Iff (Membership.mem (Sym2.pmap f z h) b) (Exists fun a => Exists fun ha => E …
  -/
  induction' z with x y
  /-
    case h
    α : Type u_1
    β : Type u_2
    P : α → Prop
    f : (a : α) → P a → β
    b : β
    x y : α
    h : ∀ (a : α), Membership.mem (Sym2.mk { fst := x, snd := y }) a → P a
    ⊢ Iff (Membership.mem (Sym2.pmap f (Sym2.mk { fst := x, snd := y }) h) b) (Exi …
  -/
  rw [pmap_pair f x y h]
  /-
    case h
    α : Type u_1
    β : Type u_2
    P : α → Prop
    f : (a : α) → P a → β
    b : β
    x y : α
    h : ∀ (a : α), Membership.mem (Sym2.mk { fst := x, snd := y }) a → P a
    ⊢ Iff (Membership.mem (Sym2.mk { fst := f x ⋯, snd := f y ⋯ }) b) (Exists fun  …
  -/
  aesop
  /-
    🎉 no goals
  -/


lemma pmap_eq_map {P : α → Prop} (f : α → β) (z : Sym2 α) (h : ∀ a ∈ z, P a) :
    z.pmap (fun a _ => f a) h = z.map f := by
  /-
    α : Type u_1
    β : Type u_2
    P : α → Prop
    f : α → β
    z : Sym2 α
    h : ∀ (a : α), Membership.mem z a → P a
    ⊢ Eq (Sym2.pmap (fun a x => f a) z h) (Sym2.map f z)
  -/
  induction' z with x y
  /-
    case h
    α : Type u_1
    β : Type u_2
    P : α → Prop
    f : α → β
    x y : α
    h : ∀ (a : α), Membership.mem (Sym2.mk { fst := x, snd := y }) a → P a
    ⊢ Eq (Sym2.pmap (fun a x => f a) (Sym2.mk { fst := x, snd := y }) h) (Sym2.map …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma map_pmap {Q : β → Prop} (f : α → β) (g : ∀ b, Q b → γ) (z : Sym2 α) (h : ∀ b ∈ z.map f, Q b):
    (z.map f).pmap g h =
    z.pmap (fun a ha => g (f a) (h (f a) (mem_map.mpr ⟨a, ha, rfl⟩))) (fun _ ha => ha) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Q : β → Prop
    f : α → β
    g : (b : β) → Q b → γ
    z : Sym2 α
    h : ∀ (b : β), Membership.mem (Sym2.map f z) b → Q b
    ⊢ Eq (Sym2.pmap g (Sym2.map f z) h) (Sym2.pmap (fun a ha => g (f a) ⋯) z ⋯)
  -/
  induction' z with x y
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Q : β → Prop
    f : α → β
    g : (b : β) → Q b → γ
    x y : α
    h : ∀ (b : β), Membership.mem (Sym2.map f (Sym2.mk { fst := x, snd := y })) b  …
    ⊢ Eq (Sym2.pmap g (Sym2.map f (Sym2.mk { fst := x, snd := y })) h) (Sym2.pmap  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma pmap_map {P : α → Prop} {Q : β → Prop} (f : ∀ a, P a → β) (g : β → γ)
    (z : Sym2 α) (h : ∀ a ∈ z, P a) (h' : ∀ b ∈ z.pmap f h, Q b) :
    (z.pmap f h).map g = z.pmap (fun a ha => g (f a (h a ha))) (fun _ ha ↦ ha) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    P : α → Prop
    Q : β → Prop
    f : (a : α) → P a → β
    g : β → γ
    z : Sym2 α
    h : ∀ (a : α), Membership.mem z a → P a
    h' : ∀ (b : β), Membership.mem (Sym2.pmap f z h) b → Q b
    ⊢ Eq (Sym2.map g (Sym2.pmap f z h)) (Sym2.pmap (fun a ha => g (f a ⋯)) z ⋯)
  -/
  induction' z with x y
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    P : α → Prop
    Q : β → Prop
    f : (a : α) → P a → β
    g : β → γ
    x y : α
    h : ∀ (a : α), Membership.mem (Sym2.mk { fst := x, snd := y }) a → P a
    h' : ∀ (b : β), Membership.mem (Sym2.pmap f (Sym2.mk { fst := x, snd := y }) h …
    ⊢ Eq (Sym2.map g (Sym2.pmap f (Sym2.mk { fst := x, snd := y }) h)) (Sym2.pmap  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma pmap_pmap {P : α → Prop} {Q : β → Prop} (f : ∀ a, P a → β) (g : ∀ b, Q b → γ)
    (z : Sym2 α) (h : ∀ a ∈ z, P a) (h' : ∀ b ∈ z.pmap f h, Q b) :
    (z.pmap f h).pmap g h' = z.pmap (fun a ha => g (f a (h a ha))
    (h' _ ((mem_pmap_iff f z h _).mpr ⟨a, ha, rfl⟩))) (fun _ ha ↦ ha) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    P : α → Prop
    Q : β → Prop
    f : (a : α) → P a → β
    g : (b : β) → Q b → γ
    z : Sym2 α
    h : ∀ (a : α), Membership.mem z a → P a
    h' : ∀ (b : β), Membership.mem (Sym2.pmap f z h) b → Q b
    ⊢ Eq (Sym2.pmap g (Sym2.pmap f z h) h') (Sym2.pmap (fun a ha => g (f a ⋯) ⋯) z …
  -/
  induction' z with x y
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    P : α → Prop
    Q : β → Prop
    f : (a : α) → P a → β
    g : (b : β) → Q b → γ
    x y : α
    h : ∀ (a : α), Membership.mem (Sym2.mk { fst := x, snd := y }) a → P a
    h' : ∀ (b : β), Membership.mem (Sym2.pmap f (Sym2.mk { fst := x, snd := y }) h …
    ⊢ Eq (Sym2.pmap g (Sym2.pmap f (Sym2.mk { fst := x, snd := y }) h) h') (Sym2.p …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma pmap_subtype_map_subtypeVal {P : α → Prop} (s : Sym2 α) (h : ∀ a ∈ s, P a) :
    (s.pmap Subtype.mk h).map Subtype.val = s := by
  /-
    α : Type u_1
    P : α → Prop
    s : Sym2 α
    h : ∀ (a : α), Membership.mem s a → P a
    ⊢ Eq (Sym2.map Subtype.val (Sym2.pmap Subtype.mk s h)) s
  -/
  induction' s with x y
  /-
    case h
    α : Type u_1
    P : α → Prop
    x y : α
    h : ∀ (a : α), Membership.mem (Sym2.mk { fst := x, snd := y }) a → P a
    ⊢ Eq (Sym2.map Subtype.val (Sym2.pmap Subtype.mk (Sym2.mk { fst := x, snd := y …
  -/
  rfl
  /-
    🎉 no goals
  -/


/--
"Attach" a proof `P a` that holds for all the elements of `s` to produce a new Sym2 object
with the same elements but in the type `{x // P x}`.
-/
def attachWith {P : α → Prop} (s : Sym2 α) (h : ∀ a ∈ s, P a) : Sym2 {a // P a} :=
  pmap Subtype.mk s h


@[simp]
lemma attachWith_map_subtypeVal {s : Sym2 α} {P : α → Prop} (h : ∀ a ∈ s, P a) :
    (s.attachWith h).map Subtype.val = s := by
  /-
    α : Type u_1
    s : Sym2 α
    P : α → Prop
    h : ∀ (a : α), Membership.mem s a → P a
    ⊢ Eq (Sym2.map Subtype.val (s.attachWith h)) s
  -/
  induction' s with x y
  /-
    case h
    α : Type u_1
    P : α → Prop
    x y : α
    h : ∀ (a : α), Membership.mem (Sym2.mk { fst := x, snd := y }) a → P a
    ⊢ Eq (Sym2.map Subtype.val ((Sym2.mk { fst := x, snd := y }).attachWith h)) (S …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A type `α` is naturally included in the diagonal of `α × α`, and this function gives the image
of this diagonal in `Sym2 α`.
-/
def diag (x : α) : Sym2 α := s(x, x)


theorem diag_injective : Function.Injective (Sym2.diag : α → Sym2 α) := fun x y h => by
  /-
    α : Type u_1
    x y : α
    h : Eq (Sym2.diag x) (Sym2.diag y)
    ⊢ Eq x y
  -/
                         /-
                           🎉 no goals
                         -/
  cases Sym2.exact h <;> rfl
                         /-
                           🎉 no goals
                         -/


/-- A predicate for testing whether an element of `Sym2 α` is on the diagonal.
-/
def IsDiag : Sym2 α → Prop :=
  lift ⟨Eq, fun _ _ => propext eq_comm⟩


theorem mk_isDiag_iff {x y : α} : IsDiag s(x, y) ↔ x = y :=
  Iff.rfl


@[simp]
theorem isDiag_iff_proj_eq (z : α × α) : IsDiag (Sym2.mk z) ↔ z.1 = z.2 :=
  Prod.recOn z fun _ _ => mk_isDiag_iff


protected lemma IsDiag.map : e.IsDiag → (e.map f).IsDiag := Sym2.ind (fun _ _ ↦ congr_arg f) e


lemma isDiag_map (hf : Injective f) : (e.map f).IsDiag ↔ e.IsDiag :=
  Sym2.ind (fun _ _ ↦ hf.eq_iff) e


@[simp]
theorem diag_isDiag (a : α) : IsDiag (diag a) :=
  Eq.refl a


theorem IsDiag.mem_range_diag {z : Sym2 α} : IsDiag z → z ∈ Set.range (@diag α) := by
  /-
    α : Type u_1
    z : Sym2 α
    ⊢ z.IsDiag → Membership.mem (Set.range Sym2.diag) z
  -/
  induction' z with x y
  /-
    case h
    α : Type u_1
    x y : α
    ⊢ (Sym2.mk { fst := x, snd := y }).IsDiag → Membership.mem (Set.range Sym2.dia …
  -/
  rintro (rfl : x = y)
  /-
    case h
    α : Type u_1
    x : α
    ⊢ Membership.mem (Set.range Sym2.diag) (Sym2.mk { fst := x, snd := x })
  -/
  exact ⟨_, rfl⟩
  /-
    🎉 no goals
  -/


theorem isDiag_iff_mem_range_diag (z : Sym2 α) : IsDiag z ↔ z ∈ Set.range (@diag α) :=
  ⟨IsDiag.mem_range_diag, fun ⟨i, hi⟩ => hi ▸ diag_isDiag i⟩


instance IsDiag.decidablePred (α : Type u) [DecidableEq α] : DecidablePred (@IsDiag α) :=
  fun z => z.recOnSubsingleton fun a => decidable_of_iff' _ (isDiag_iff_proj_eq a)


theorem other_ne {a : α} {z : Sym2 α} (hd : ¬IsDiag z) (h : a ∈ z) : Mem.other h ≠ a := by
  /-
    α : Type u_1
    a : α
    z : Sym2 α
    hd : Not z.IsDiag
    h : Membership.mem z a
    ⊢ Ne (Sym2.Mem.other h) a
  -/
  contrapose! hd
  /-
    α : Type u_1
    a : α
    z : Sym2 α
    h : Membership.mem z a
    hd : Eq (Sym2.Mem.other h) a
    ⊢ z.IsDiag
  -/
  have h' := Sym2.other_spec h
  /-
    α : Type u_1
    a : α
    z : Sym2 α
    h : Membership.mem z a
    hd : Eq (Sym2.Mem.other h) a
    h' : Eq (Sym2.mk { fst := a, snd := Sym2.Mem.other h }) z
    ⊢ z.IsDiag
  -/
  rw [hd] at h'
  /-
    α : Type u_1
    a : α
    z : Sym2 α
    h : Membership.mem z a
    hd : Eq (Sym2.Mem.other h) a
    h' : Eq (Sym2.mk { fst := a, snd := a }) z
    ⊢ z.IsDiag
  -/
  rw [← h']
  /-
    α : Type u_1
    a : α
    z : Sym2 α
    h : Membership.mem z a
    hd : Eq (Sym2.Mem.other h) a
    h' : Eq (Sym2.mk { fst := a, snd := a }) z
    ⊢ (Sym2.mk { fst := a, snd := a }).IsDiag
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Symmetric relations define a set on `Sym2 α` by taking all those pairs
of elements that are related.
-/
def fromRel (sym : Symmetric r) : Set (Sym2 α) :=
  setOf (lift ⟨r, fun _ _ => propext ⟨(sym ·), (sym ·)⟩⟩)


@[simp]
theorem fromRel_proj_prop {sym : Symmetric r} {z : α × α} : Sym2.mk z ∈ fromRel sym ↔ r z.1 z.2 :=
  Iff.rfl


theorem fromRel_prop {sym : Symmetric r} {a b : α} : s(a, b) ∈ fromRel sym ↔ r a b :=
  Iff.rfl


theorem fromRel_bot : fromRel (fun (_ _ : α) z => z : Symmetric ⊥) = ∅ := by
  /-
    α : Type u_1
    ⊢ Eq (Sym2.fromRel ⋯) EmptyCollection.emptyCollection
  -/
  apply Set.eq_empty_of_forall_not_mem fun e => _
  /-
    α : Type u_1
    ⊢ ∀ (e : Sym2 α), Not (Membership.mem (Sym2.fromRel ⋯) e)
  -/
  apply Sym2.ind
  /-
    case h
    α : Type u_1
    ⊢ ∀ (x y : α), Not (Membership.mem (Sym2.fromRel ⋯) (Sym2.mk { fst := x, snd : …
  -/
  simp [-Set.bot_eq_empty, Prop.bot_eq_false]
  /-
    🎉 no goals
  -/


theorem fromRel_top : fromRel (fun (_ _ : α) z => z : Symmetric ⊤) = Set.univ := by
  /-
    α : Type u_1
    ⊢ Eq (Sym2.fromRel ⋯) Set.univ
  -/
  apply Set.eq_univ_of_forall fun e => _
  /-
    α : Type u_1
    ⊢ ∀ (e : Sym2 α), Membership.mem (Sym2.fromRel ⋯) e
  -/
  apply Sym2.ind
  /-
    case h
    α : Type u_1
    ⊢ ∀ (x y : α), Membership.mem (Sym2.fromRel ⋯) (Sym2.mk { fst := x, snd := y })
  -/
  simp [-Set.top_eq_univ, Prop.top_eq_true]
  /-
    🎉 no goals
  -/


theorem fromRel_ne : fromRel (fun (_ _ : α) z => z.symm : Symmetric Ne) = {z | ¬IsDiag z} := by
  /-
    α : Type u_1
    ⊢ Eq (Sym2.fromRel ⋯) (setOf fun z => Not z.IsDiag)
  -/
  ext z; exact z.ind (by simp)
         /-
           🎉 no goals
         -/


theorem fromRel_irreflexive {sym : Symmetric r} :
    Irreflexive r ↔ ∀ {z}, z ∈ fromRel sym → ¬IsDiag z :=
             /-
               α : Type u_1
               r : α → α → Prop
               sym : Symmetric r
               ⊢ Irreflexive r → ∀ {z : Sym2 α}, Membership.mem (Sym2.fromRel sym) z → Not z. …
             -/
  { mp := by intro h; apply Sym2.ind; aesop
                                      /-
                                        🎉 no goals
                                      -/
    mpr := fun h _ hr => h (fromRel_prop.mpr hr) rfl }


theorem mem_fromRel_irrefl_other_ne {sym : Symmetric r} (irrefl : Irreflexive r) {a : α}
    {z : Sym2 α} (hz : z ∈ fromRel sym) (h : a ∈ z) : Mem.other h ≠ a :=
  other_ne (fromRel_irreflexive.mp irrefl hz) h


instance fromRel.decidablePred (sym : Symmetric r) [h : DecidableRel r] :
    DecidablePred (· ∈ Sym2.fromRel sym) := fun z => z.recOnSubsingleton fun _ => h _ _


/-- The inverse to `Sym2.fromRel`. Given a set on `Sym2 α`, give a symmetric relation on `α`
(see `Sym2.toRel_symmetric`). -/
def ToRel (s : Set (Sym2 α)) (x y : α) : Prop :=
  s(x, y) ∈ s


@[simp]
theorem toRel_prop (s : Set (Sym2 α)) (x y : α) : ToRel s x y ↔ s(x, y) ∈ s :=
  Iff.rfl


                                                                                  /-
                                                                                    α : Type u_1
                                                                                    s : Set (Sym2 α)
                                                                                    x y : α
                                                                                    ⊢ Sym2.ToRel s x y → Sym2.ToRel s y x
                                                                                  -/
theorem toRel_symmetric (s : Set (Sym2 α)) : Symmetric (ToRel s) := fun x y => by simp [eq_swap]
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem toRel_fromRel (sym : Symmetric r) : ToRel (fromRel sym) = r :=
  rfl


theorem fromRel_toRel (s : Set (Sym2 α)) : fromRel (toRel_symmetric s) = s :=
  Set.ext fun z => Sym2.ind (fun _ _ => Iff.rfl) z


private def fromVector : List.Vector α 2 → α × α
  | ⟨[a, b], _⟩ => (a, b)


private theorem perm_card_two_iff {a₁ b₁ a₂ b₂ : α} :
    [a₁, b₁].Perm [a₂, b₂] ↔ a₁ = a₂ ∧ b₁ = b₂ ∨ a₁ = b₂ ∧ b₁ = a₂ :=
  { mp := by
      simp only [← Multiset.coe_eq_coe, ← Multiset.cons_coe, Multiset.coe_nil, Multiset.cons_zero,
        Multiset.cons_eq_cons, Multiset.singleton_inj, ne_eq, Multiset.singleton_eq_cons_iff,
        exists_eq_right_right, and_true]
      /-
        α : Type u_1
        a₁ b₁ a₂ b₂ : α
        ⊢ Or (And (Eq a₁ a₂) (Eq b₁ b₂)) (And (Not (Eq a₁ a₂)) (And (Eq b₁ a₂) (Eq b₂  …
      -/
      tauto
      /-
        🎉 no goals
      -/
    mpr := fun
        | .inl ⟨h₁, h₂⟩ | .inr ⟨h₁, h₂⟩ => by
          /-
            α : Type u_1
            a₁ b₁ a₂ b₂ : α
            h₁ : Eq a₁ a₂
            h₂ : Eq b₁ b₂
            ⊢ (List.cons a₁ (List.cons b₁ List.nil)).Perm (List.cons a₂ (List.cons b₂ List …
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
            α : Type u_1
            a₁ b₁ a₂ b₂ : α
            h₁ : Eq a₁ b₂
            h₂ : Eq b₁ a₂
            ⊢ (List.cons b₂ (List.cons a₂ List.nil)).Perm (List.cons a₂ (List.cons b₂ List …
          -/
          first | done | apply List.Perm.swap'; rfl }
          /-
            🎉 no goals
          -/


/-- The symmetric square is equivalent to length-2 vectors up to permutations. -/
def sym2EquivSym' : Equiv (Sym2 α) (Sym' α 2) where
  toFun :=
    Quot.map (fun x : α × α => ⟨[x.1, x.2], rfl⟩)
      (by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          ⊢ ∀ ⦃a b : Prod α α⦄, Sym2.Rel α a b → (List.Vector.Perm.isSetoid α 2) ((fun x …
        -/
        rintro _ _ ⟨_⟩
          /-
            case refl
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            e : Sym2 α
            f : α → β
            x✝ y✝ : α
            ⊢ (List.Vector.Perm.isSetoid α 2) ((fun x => ⟨List.cons x.1 (List.cons x.2 Lis …
          -/
        · constructor; apply List.Perm.refl
                       /-
                         🎉 no goals
                       -/
        /-
          case swap
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          x✝ y✝ : α
          ⊢ (List.Vector.Perm.isSetoid α 2) ((fun x => ⟨List.cons x.1 (List.cons x.2 Lis …
        -/
        apply List.Perm.swap'
        /-
          case swap.p
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          x✝ y✝ : α
          ⊢ List.nil.Perm List.nil
        -/
        rfl)
        /-
          🎉 no goals
        -/
  invFun :=
    Quot.map fromVector
      (by
        /-
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          ⊢ ∀ ⦃a b : List.Vector α 2⦄, (List.Vector.Perm.isSetoid α 2) a b → Sym2.Rel α  …
        -/
        rintro ⟨x, hx⟩ ⟨y, hy⟩ h
        /-
          case mk.mk
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          x : List α
          hx : Eq x.length 2
          y : List α
          hy : Eq y.length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨x, hx⟩ ⟨y, hy⟩
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨x, hx⟩) (Sym2.fromVector ⟨y, hy⟩)
        -/
        cases' x with _ x; · simp at hx
                             /-
                               🎉 no goals
                             -/
        /-
          case mk.mk.cons
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          y : List α
          hy : Eq y.length 2
          head✝ : α
          x : List α
          hx : Eq (List.cons head✝ x).length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝ x, hx⟩ ⟨y, hy⟩
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝ x, hx⟩) (Sym2.fromVector ⟨y, hy⟩)
        -/
        cases' x with _ x; · simp at hx
                             /-
                               🎉 no goals
                             -/
        /-
          case mk.mk.cons.cons
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          y : List α
          hy : Eq y.length 2
          head✝¹ head✝ : α
          x : List α
          hx : Eq (List.cons head✝¹ (List.cons head✝ x)).length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝¹ (List.cons head✝ x), hx⟩ …
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝¹ (List.cons head✝ x), hx⟩) (Sym …
        -/
        cases' x with _ x; swap
          /-
            case mk.mk.cons.cons.cons
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            e : Sym2 α
            f : α → β
            y : List α
            hy : Eq y.length 2
            head✝² head✝¹ head✝ : α
            x : List α
            hx : Eq (List.cons head✝² (List.cons head✝¹ (List.cons head✝ x))).length 2
            h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝² (List.cons head✝¹ (List. …
            ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝² (List.cons head✝¹ (List.cons h …
          -/
        · exfalso
          /-
            case mk.mk.cons.cons.cons
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            e : Sym2 α
            f : α → β
            y : List α
            hy : Eq y.length 2
            head✝² head✝¹ head✝ : α
            x : List α
            hx : Eq (List.cons head✝² (List.cons head✝¹ (List.cons head✝ x))).length 2
            h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝² (List.cons head✝¹ (List. …
            ⊢ False
          -/
          simp at hx
          /-
            🎉 no goals
          -/
        /-
          case mk.mk.cons.cons.nil
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          y : List α
          hy : Eq y.length 2
          head✝¹ head✝ : α
          hx : Eq (List.cons head✝¹ (List.cons head✝ List.nil)).length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝¹ (List.cons head✝ List.ni …
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝¹ (List.cons head✝ List.nil), hx …
        -/
        cases' y with _ y; · simp at hy
                             /-
                               🎉 no goals
                             -/
        /-
          case mk.mk.cons.cons.nil.cons
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          head✝² head✝¹ : α
          hx : Eq (List.cons head✝² (List.cons head✝¹ List.nil)).length 2
          head✝ : α
          y : List α
          hy : Eq (List.cons head✝ y).length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝² (List.cons head✝¹ List.n …
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝² (List.cons head✝¹ List.nil), h …
        -/
        cases' y with _ y; · simp at hy
                             /-
                               🎉 no goals
                             -/
        /-
          case mk.mk.cons.cons.nil.cons.cons
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          head✝³ head✝² : α
          hx : Eq (List.cons head✝³ (List.cons head✝² List.nil)).length 2
          head✝¹ head✝ : α
          y : List α
          hy : Eq (List.cons head✝¹ (List.cons head✝ y)).length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝³ (List.cons head✝² List.n …
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝³ (List.cons head✝² List.nil), h …
        -/
        cases' y with _ y; swap
          /-
            case mk.mk.cons.cons.nil.cons.cons.cons
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            e : Sym2 α
            f : α → β
            head✝⁴ head✝³ : α
            hx : Eq (List.cons head✝⁴ (List.cons head✝³ List.nil)).length 2
            head✝² head✝¹ head✝ : α
            y : List α
            hy : Eq (List.cons head✝² (List.cons head✝¹ (List.cons head✝ y))).length 2
            h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝⁴ (List.cons head✝³ List.n …
            ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝⁴ (List.cons head✝³ List.nil), h …
          -/
        · exfalso
          /-
            case mk.mk.cons.cons.nil.cons.cons.cons
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            e : Sym2 α
            f : α → β
            head✝⁴ head✝³ : α
            hx : Eq (List.cons head✝⁴ (List.cons head✝³ List.nil)).length 2
            head✝² head✝¹ head✝ : α
            y : List α
            hy : Eq (List.cons head✝² (List.cons head✝¹ (List.cons head✝ y))).length 2
            h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝⁴ (List.cons head✝³ List.n …
            ⊢ False
          -/
          simp at hy
          /-
            🎉 no goals
          -/
        /-
          case mk.mk.cons.cons.nil.cons.cons.nil
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          head✝³ head✝² : α
          hx : Eq (List.cons head✝³ (List.cons head✝² List.nil)).length 2
          head✝¹ head✝ : α
          hy : Eq (List.cons head✝¹ (List.cons head✝ List.nil)).length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝³ (List.cons head✝² List.n …
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝³ (List.cons head✝² List.nil), h …
        -/
        rcases perm_card_two_iff.mp h with (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)
          /-
            case mk.mk.cons.cons.nil.cons.cons.nil.inl.intro
            α : Type u_1
            β : Type u_2
            γ : Type u_3
            e : Sym2 α
            f : α → β
            head✝¹ head✝ : α
            hx hy : Eq (List.cons head✝¹ (List.cons head✝ List.nil)).length 2
            h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝¹ (List.cons head✝ List.ni …
            ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝¹ (List.cons head✝ List.nil), hx …
          -/
        · constructor
          /-
            🎉 no goals
          -/
        /-
          case mk.mk.cons.cons.nil.cons.cons.nil.inr.intro
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          e : Sym2 α
          f : α → β
          head✝¹ head✝ : α
          hx : Eq (List.cons head✝¹ (List.cons head✝ List.nil)).length 2
          hy : Eq (List.cons head✝ (List.cons head✝¹ List.nil)).length 2
          h : (List.Vector.Perm.isSetoid α 2) ⟨List.cons head✝¹ (List.cons head✝ List.ni …
          ⊢ Sym2.Rel α (Sym2.fromVector ⟨List.cons head✝¹ (List.cons head✝ List.nil), hx …
        -/
        apply Sym2.Rel.swap)
        /-
          🎉 no goals
        -/
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   e : Sym2 α
                   f : α → β
                   ⊢ Function.LeftInverse (Quot.map Sym2.fromVector ⋯) (Quot.map (fun x => ⟨List. …
                 -/
  left_inv := by apply Sym2.ind; aesop (add norm unfold [Sym2.fromVector])
                                 /-
                                   🎉 no goals
                                 -/
  right_inv x := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      x : Sym.Sym' α 2
      ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
    -/
    refine x.recOnSubsingleton fun x => ?_
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      x✝ : Sym.Sym' α 2
      x : List.Vector α 2
      ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
    -/
    cases' x with x hx
    /-
      case mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      x✝ : Sym.Sym' α 2
      x : List α
      hx : Eq x.length 2
      ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
    -/
    cases' x with _ x
      /-
        case mk.nil
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        e : Sym2 α
        f : α → β
        x : Sym.Sym' α 2
        hx : Eq List.nil.length 2
        ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
      -/
    · simp at hx
      /-
        🎉 no goals
      -/
    /-
      case mk.cons
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      x✝ : Sym.Sym' α 2
      head✝ : α
      x : List α
      hx : Eq (List.cons head✝ x).length 2
      ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
    -/
    cases' x with _ x
      /-
        case mk.cons.nil
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        e : Sym2 α
        f : α → β
        x : Sym.Sym' α 2
        head✝ : α
        hx : Eq (List.cons head✝ List.nil).length 2
        ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
      -/
    · simp at hx
      /-
        🎉 no goals
      -/
    /-
      case mk.cons.cons
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      x✝ : Sym.Sym' α 2
      head✝¹ head✝ : α
      x : List α
      hx : Eq (List.cons head✝¹ (List.cons head✝ x)).length 2
      ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
    -/
    cases' x with _ x
    /-
      case mk.cons.cons.nil
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      x : Sym.Sym' α 2
      head✝¹ head✝ : α
      hx : Eq (List.cons head✝¹ (List.cons head✝ List.nil)).length 2
      ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
    -/
    swap
      /-
        case mk.cons.cons.cons
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        e : Sym2 α
        f : α → β
        x✝ : Sym.Sym' α 2
        head✝² head✝¹ head✝ : α
        x : List α
        hx : Eq (List.cons head✝² (List.cons head✝¹ (List.cons head✝ x))).length 2
        ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
      -/
    · exfalso
      /-
        case mk.cons.cons.cons
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        e : Sym2 α
        f : α → β
        x✝ : Sym.Sym' α 2
        head✝² head✝¹ head✝ : α
        x : List α
        hx : Eq (List.cons head✝² (List.cons head✝¹ (List.cons head✝ x))).length 2
        ⊢ False
      -/
      simp at hx
      /-
        🎉 no goals
      -/
    /-
      case mk.cons.cons.nil
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      x : Sym.Sym' α 2
      head✝¹ head✝ : α
      hx : Eq (List.cons head✝¹ (List.cons head✝ List.nil)).length 2
      ⊢ Eq (Quot.map (fun x => ⟨List.cons x.1 (List.cons x.2 List.nil), ⋯⟩) ⋯ (Quot. …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The symmetric square is equivalent to the second symmetric power. -/
def equivSym (α : Type*) : Sym2 α ≃ Sym α 2 :=
  Equiv.trans sym2EquivSym' symEquivSym'.symm


/-- The symmetric square is equivalent to multisets of cardinality
two. (This is currently a synonym for `equivSym`, but it's provided
in case the definition for `Sym` changes.) -/
def equivMultiset (α : Type*) : Sym2 α ≃ { s : Multiset α // Multiset.card s = 2 } :=
  equivSym α


/-- Given `[DecidableEq α]` and `[Fintype α]`, the following instance gives `Fintype (Sym2 α)`.
-/
instance instDecidableRel [DecidableEq α] : DecidableRel (Rel α) :=
  fun _ _ => decidable_of_iff' _ rel_iff


instance instDecidableRel' [DecidableEq α] : DecidableRel (HasEquiv.Equiv (α := α × α)) :=
  instDecidableRel


instance [DecidableEq α] : DecidableEq (Sym2 α) :=
  inferInstanceAs <| DecidableEq (Quotient (Sym2.Rel.setoid α))


/--
A function that gives the other element of a pair given one of the elements.  Used in `Mem.other'`.
-/
@[aesop norm unfold (rule_sets := [Sym2])]
private def pairOther [DecidableEq α] (a : α) (z : α × α) : α :=
  if a = z.1 then z.2 else z.1



/-- Get the other element of the unordered pair using the decidable equality.
This is the computable version of `Mem.other`. -/
@[aesop norm unfold (rule_sets := [Sym2])]
def Mem.other' [DecidableEq α] {a : α} {z : Sym2 α} (h : a ∈ z) : α :=
  Sym2.rec (fun s _ => pairOther a s) (by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      inst✝ : DecidableEq α
      a : α
      z : Sym2 α
      h : Membership.mem z a
      ⊢ ∀ (p q : Prod α α) (h : Sym2.Rel α p q), Eq (Eq.ndrec (motive := fun x => Me …
    -/
    clear h z
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      inst✝ : DecidableEq α
      a : α
      ⊢ ∀ (p q : Prod α α) (h : Sym2.Rel α p q), Eq (Eq.ndrec (motive := fun x => Me …
    -/
    intro x y h
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      inst✝ : DecidableEq α
      a : α
      x y : Prod α α
      h : Sym2.Rel α x y
      ⊢ Eq (Eq.ndrec (motive := fun x => Membership.mem x a → α) (fun x_1 => Sym2.pa …
    -/
    ext hy
    /-
      case h
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      e : Sym2 α
      f : α → β
      inst✝ : DecidableEq α
      a : α
      x y : Prod α α
      h : Sym2.Rel α x y
      hy : Membership.mem (Sym2.mk y) a
      ⊢ Eq (Eq.ndrec (motive := fun x => Membership.mem x a → α) (fun x_1 => Sym2.pa …
    -/
    convert_to Sym2.pairOther a x = _
    · have : ∀ {c e h}, @Eq.ndrec (Sym2 α) (Sym2.mk x)
          (fun x => a ∈ x → α) (fun _ => Sym2.pairOther a x) c e h = Sym2.pairOther a x := by
          intro _ e _; subst e; rfl
      /-
        case h.e'_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        e : Sym2 α
        f : α → β
        inst✝ : DecidableEq α
        a : α
        x y : Prod α α
        h : Sym2.Rel α x y
        hy : Membership.mem (Sym2.mk y) a
        this : ∀ {c : Sym2 α} {e : Eq (Sym2.mk x) c} {h : Membership.mem c a}, Eq (Eq. …
        ⊢ Eq (Eq.ndrec (motive := fun x => Membership.mem x a → α) (fun x_1 => Sym2.pa …
      -/
      apply this
      /-
        🎉 no goals
      -/
      /-
        case h.convert_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        e : Sym2 α
        f : α → β
        inst✝ : DecidableEq α
        a : α
        x y : Prod α α
        h : Sym2.Rel α x y
        hy : Membership.mem (Sym2.mk y) a
        ⊢ Eq (Sym2.pairOther a x) (Sym2.pairOther a y)
      -/
    · rw [mem_iff] at hy
      /-
        case h.convert_2
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        e : Sym2 α
        f : α → β
        inst✝ : DecidableEq α
        a : α
        x y : Prod α α
        h : Sym2.Rel α x y
        hy : Or (Eq a y.1) (Eq a y.2)
        ⊢ Eq (Sym2.pairOther a x) (Sym2.pairOther a y)
      -/
      aesop (add norm unfold [pairOther]))
      /-
        🎉 no goals
      -/
    z h


@[simp]
theorem other_spec' [DecidableEq α] {a : α} {z : Sym2 α} (h : a ∈ z) : s(a, Mem.other' h) = z := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    z : Sym2 α
    h : Membership.mem z a
    ⊢ Eq (Sym2.mk { fst := a, snd := Sym2.Mem.other' h }) z
  -/
  induction z
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a x✝ y✝ : α
    h : Membership.mem (Sym2.mk { fst := x✝, snd := y✝ }) a
    ⊢ Eq (Sym2.mk { fst := a, snd := Sym2.Mem.other' h }) (Sym2.mk { fst := x✝, sn …
  -/
  have h' := mem_iff.mp h
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a x✝ y✝ : α
    h : Membership.mem (Sym2.mk { fst := x✝, snd := y✝ }) a
    h' : Or (Eq a x✝) (Eq a y✝)
    ⊢ Eq (Sym2.mk { fst := a, snd := Sym2.Mem.other' h }) (Sym2.mk { fst := x✝, sn …
  -/
  aesop (add norm unfold [Sym2.rec, Quot.rec]) (rule_sets := [Sym2])
  /-
    🎉 no goals
  -/


@[simp]
theorem other_eq_other' [DecidableEq α] {a : α} {z : Sym2 α} (h : a ∈ z) :
                                     /-
                                       α : Type u_1
                                       inst✝ : DecidableEq α
                                       a : α
                                       z : Sym2 α
                                       h : Membership.mem z a
                                       ⊢ Eq (Sym2.Mem.other h) (Sym2.Mem.other' h)
                                     -/
    Mem.other h = Mem.other' h := by rw [← congr_right, other_spec' h, other_spec]
                                     /-
                                       🎉 no goals
                                     -/


theorem other_mem' [DecidableEq α] {a : α} {z : Sym2 α} (h : a ∈ z) : Mem.other' h ∈ z := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    z : Sym2 α
    h : Membership.mem z a
    ⊢ Membership.mem z (Sym2.Mem.other' h)
  -/
  rw [← other_eq_other']
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    z : Sym2 α
    h : Membership.mem z a
    ⊢ Membership.mem z (Sym2.Mem.other h)
  -/
  exact other_mem h
  /-
    🎉 no goals
  -/


theorem other_invol' [DecidableEq α] {a : α} {z : Sym2 α} (ha : a ∈ z) (hb : Mem.other' ha ∈ z) :
    Mem.other' hb = a := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    z : Sym2 α
    ha : Membership.mem z a
    hb : Membership.mem z (Sym2.Mem.other' ha)
    ⊢ Eq (Sym2.Mem.other' hb) a
  -/
  induction z
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    a x✝ y✝ : α
    ha : Membership.mem (Sym2.mk { fst := x✝, snd := y✝ }) a
    hb : Membership.mem (Sym2.mk { fst := x✝, snd := y✝ }) (Sym2.Mem.other' ha)
    ⊢ Eq (Sym2.Mem.other' hb) a
  -/
  aesop (rule_sets := [Sym2]) (add norm unfold [Sym2.rec, Quot.rec])
  /-
    🎉 no goals
  -/


theorem other_invol {a : α} {z : Sym2 α} (ha : a ∈ z) (hb : Mem.other ha ∈ z) :
    Mem.other hb = a := by
  classical
    rw [other_eq_other'] at hb ⊢
    convert other_invol' ha hb using 2
    apply other_eq_other'


theorem filter_image_mk_isDiag [DecidableEq α] (s : Finset α) :
    {a ∈ (s ×ˢ s).image Sym2.mk | a.IsDiag} = s.diag.image Sym2.mk := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Finset.filter (fun a => a.IsDiag) (Finset.image Sym2.mk (SProd.sprod s s …
  -/
  ext z
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    z : Sym2 α
    ⊢ Iff (Membership.mem (Finset.filter (fun a => a.IsDiag) (Finset.image Sym2.mk …
  -/
  induction' z
  /-
    case h.h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x✝ y✝ : α
    ⊢ Iff (Membership.mem (Finset.filter (fun a => a.IsDiag) (Finset.image Sym2.mk …
  -/
  simp only [mem_image, mem_diag, exists_prop, mem_filter, Prod.exists, mem_product]
  /-
    case h.h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x✝ y✝ : α
    ⊢ Iff (And (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Mem …
  -/
  constructor
    /-
      case h.h.mp
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ : α
      ⊢ And (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membersh …
    -/
  · rintro ⟨⟨a, b, ⟨ha, hb⟩, h⟩, hab⟩
    /-
      case h.h.mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ : α
      hab : (Sym2.mk { fst := x✝, snd := y✝ }).IsDiag
      a b : α
      h : Eq (Sym2.mk { fst := a, snd := b }) (Sym2.mk { fst := x✝, snd := y✝ })
      ha : Membership.mem s a
      hb : Membership.mem s b
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (Eq a b)) (Eq  …
    -/
    rw [← h, Sym2.mk_isDiag_iff] at hab
    /-
      case h.h.mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ a b : α
      hab : Eq a b
      h : Eq (Sym2.mk { fst := a, snd := b }) (Sym2.mk { fst := x✝, snd := y✝ })
      ha : Membership.mem s a
      hb : Membership.mem s b
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (Eq a b)) (Eq  …
    -/
    exact ⟨a, b, ⟨ha, hab⟩, h⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ : α
      ⊢ (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Eq a b)) (Eq …
    -/
  · rintro ⟨a, b, ⟨ha, rfl⟩, h⟩
    /-
      case h.h.mpr.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ a : α
      ha : Membership.mem s a
      h : Eq (Sym2.mk { fst := a, snd := a }) (Sym2.mk { fst := x✝, snd := y✝ })
      ⊢ And (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membersh …
    -/
    rw [← h]
    /-
      case h.h.mpr.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ a : α
      ha : Membership.mem s a
      h : Eq (Sym2.mk { fst := a, snd := a }) (Sym2.mk { fst := x✝, snd := y✝ })
      ⊢ And (Exists fun a_1 => Exists fun b => And (And (Membership.mem s a_1) (Memb …
    -/
    exact ⟨⟨a, a, ⟨ha, ha⟩, rfl⟩, rfl⟩
    /-
      🎉 no goals
    -/


theorem filter_image_mk_not_isDiag [DecidableEq α] (s : Finset α) :
    {a ∈ (s ×ˢ s).image Sym2.mk | ¬a.IsDiag} = s.offDiag.image Sym2.mk := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Finset.filter (fun a => Not a.IsDiag) (Finset.image Sym2.mk (SProd.sprod …
  -/
  ext z
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    z : Sym2 α
    ⊢ Iff (Membership.mem (Finset.filter (fun a => Not a.IsDiag) (Finset.image Sym …
  -/
  induction z
  /-
    case h.h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x✝ y✝ : α
    ⊢ Iff (Membership.mem (Finset.filter (fun a => Not a.IsDiag) (Finset.image Sym …
  -/
  simp only [mem_image, mem_offDiag, mem_filter, Prod.exists, mem_product]
  /-
    case h.h
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x✝ y✝ : α
    ⊢ Iff (And (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Mem …
  -/
  constructor
    /-
      case h.h.mp
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ : α
      ⊢ And (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membersh …
    -/
  · rintro ⟨⟨a, b, ⟨ha, hb⟩, h⟩, hab⟩
    /-
      case h.h.mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ : α
      hab : Not (Sym2.mk { fst := x✝, snd := y✝ }).IsDiag
      a b : α
      h : Eq (Sym2.mk { fst := a, snd := b }) (Sym2.mk { fst := x✝, snd := y✝ })
      ha : Membership.mem s a
      hb : Membership.mem s b
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (And (Membersh …
    -/
    rw [← h, Sym2.mk_isDiag_iff] at hab
    /-
      case h.h.mp.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ a b : α
      hab : Not (Eq a b)
      h : Eq (Sym2.mk { fst := a, snd := b }) (Sym2.mk { fst := x✝, snd := y✝ })
      ha : Membership.mem s a
      hb : Membership.mem s b
      ⊢ Exists fun a => Exists fun b => And (And (Membership.mem s a) (And (Membersh …
    -/
    exact ⟨a, b, ⟨ha, hb, hab⟩, h⟩
    /-
      🎉 no goals
    -/
    /-
      case h.h.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ : α
      ⊢ (Exists fun a => Exists fun b => And (And (Membership.mem s a) (And (Members …
    -/
  · rintro ⟨a, b, ⟨ha, hb, hab⟩, h⟩
    /-
      case h.h.mpr.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ a b : α
      h : Eq (Sym2.mk { fst := a, snd := b }) (Sym2.mk { fst := x✝, snd := y✝ })
      ha : Membership.mem s a
      hb : Membership.mem s b
      hab : Ne a b
      ⊢ And (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membersh …
    -/
    rw [Ne, ← Sym2.mk_isDiag_iff, h] at hab
    /-
      case h.h.mpr.intro.intro.intro.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      x✝ y✝ a b : α
      h : Eq (Sym2.mk { fst := a, snd := b }) (Sym2.mk { fst := x✝, snd := y✝ })
      ha : Membership.mem s a
      hb : Membership.mem s b
      hab : Not (Sym2.mk { fst := x✝, snd := y✝ }).IsDiag
      ⊢ And (Exists fun a => Exists fun b => And (And (Membership.mem s a) (Membersh …
    -/
    exact ⟨⟨a, b, ⟨ha, hb⟩, h⟩, hab⟩
    /-
      🎉 no goals
    -/


instance [Subsingleton α] : Subsingleton (Sym2 α) :=
  (equivSym α).injective.subsingleton


instance [Unique α] : Unique (Sym2 α) :=
  Unique.mk' _


instance [IsEmpty α] : IsEmpty (Sym2 α) :=
  (equivSym α).isEmpty


instance [Nontrivial α] : Nontrivial (Sym2 α) :=
  diag_injective.nontrivial

-- TODO: use a sort order if available, https://github.com/leanprover-community/mathlib/issues/18166

unsafe instance [Repr α] : Repr (Sym2 α) where
  reprPrec s _ := f!"s({repr s.unquot.1}, {repr s.unquot.2})"


