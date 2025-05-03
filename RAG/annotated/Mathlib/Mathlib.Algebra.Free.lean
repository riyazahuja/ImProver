/-- Free nonabelian additive magma over a given alphabet. -/
inductive FreeAddMagma (α : Type u) : Type u
  | of : α → FreeAddMagma α
  | add : FreeAddMagma α → FreeAddMagma α → FreeAddMagma α
  deriving DecidableEq

compile_inductive% FreeAddMagma


/-- Free magma over a given alphabet. -/
@[to_additive]
inductive FreeMagma (α : Type u) : Type u
  | of : α → FreeMagma α
  | mul : FreeMagma α → FreeMagma α → FreeMagma α
  deriving DecidableEq

compile_inductive% FreeMagma


@[to_additive]
instance [Inhabited α] : Inhabited (FreeMagma α) := ⟨of default⟩


@[to_additive]
instance : Mul (FreeMagma α) := ⟨FreeMagma.mul⟩

-- Porting note: invalid attribute 'match_pattern', declaration is in an imported module
-- attribute [match_pattern] Mul.mul


@[to_additive (attr := simp)]
theorem mul_eq (x y : FreeMagma α) : mul x y = x * y := rfl


/-- Recursor for `FreeMagma` using `x * y` instead of `FreeMagma.mul x y`. -/
@[to_additive (attr := elab_as_elim, induction_eliminator)
  "Recursor for `FreeAddMagma` using `x + y` instead of `FreeAddMagma.add x y`."]
def recOnMul {C : FreeMagma α → Sort l} (x) (ih1 : ∀ x, C (of x))
    (ih2 : ∀ x y, C x → C y → C (x * y)) : C x :=
  FreeMagma.recOn x ih1 ih2


@[to_additive (attr := ext 1100)]
theorem hom_ext {β : Type v} [Mul β] {f g : FreeMagma α →ₙ* β} (h : f ∘ of = g ∘ of) : f = g :=
                                                            /-
                                                              α : Type u
                                                              β : Type v
                                                              inst✝ : Mul β
                                                              f g : MulHom (FreeMagma α) β
                                                              h : Eq (Function.comp (⇑f) FreeMagma.of) (Function.comp (⇑g) FreeMagma.of)
                                                              x : FreeMagma α
                                                              ⊢ ∀ (x y : FreeMagma α), Eq (f x) (g x) → Eq (f y) (g y) → Eq (f (HMul.hMul x  …
                                                            -/
  (DFunLike.ext _ _) fun x ↦ recOnMul x (congr_fun h) <| by intros; simp only [map_mul, *]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- Lifts a function `α → β` to a magma homomorphism `FreeMagma α → β` given a magma `β`. -/
def FreeMagma.liftAux {α : Type u} {β : Type v} [Mul β] (f : α → β) : FreeMagma α → β
  | FreeMagma.of x => f x
  | mul x y => liftAux f x * liftAux f y


/-- Lifts a function `α → β` to an additive magma homomorphism `FreeAddMagma α → β` given
an additive magma `β`. -/
def FreeAddMagma.liftAux {α : Type u} {β : Type v} [Add β] (f : α → β) : FreeAddMagma α → β
  | FreeAddMagma.of x => f x
  | x + y => liftAux f x + liftAux f y


attribute [to_additive existing] FreeMagma.liftAux


/-- The universal property of the free magma expressing its adjointness. -/
@[to_additive (attr := simps symm_apply)
"The universal property of the free additive magma expressing its adjointness."]
def lift : (α → β) ≃ (FreeMagma α →ₙ* β) where
  toFun f :=
  { toFun := liftAux f
    map_mul' := fun _ _ ↦ rfl }
  invFun F := F ∘ of
  left_inv _ := rfl
                    /-
                      α : Type u
                      β : Type v
                      inst✝ : Mul β
                      f : α → β
                      F : MulHom (FreeMagma α) β
                      ⊢ Eq ((fun f => { toFun := FreeMagma.liftAux f, map_mul' := ⋯ }) ((fun F => Fu …
                    -/
  right_inv F := by ext; rfl
                         /-
                           🎉 no goals
                         -/


@[to_additive (attr := simp)]
theorem lift_of (x) : lift f (of x) = f x := rfl


@[to_additive (attr := simp)]
theorem lift_comp_of : lift f ∘ of = f := rfl


@[to_additive (attr := simp)]
theorem lift_comp_of' (f : FreeMagma α →ₙ* β) : lift (f ∘ of) = f := lift.apply_symm_apply f


/-- The unique magma homomorphism `FreeMagma α →ₙ* FreeMagma β` that sends
each `of x` to `of (f x)`. -/
@[to_additive "The unique additive magma homomorphism `FreeAddMagma α → FreeAddMagma β` that sends
each `of x` to `of (f x)`."]
def map (f : α → β) : FreeMagma α →ₙ* FreeMagma β := lift (of ∘ f)


@[to_additive (attr := simp)]
theorem map_of (x) : map f (of x) = of (f x) := rfl


@[to_additive]
instance : Monad FreeMagma where
  pure := of
  bind x f := lift f x


/-- Recursor on `FreeMagma` using `pure` instead of `of`. -/
@[to_additive (attr := elab_as_elim) "Recursor on `FreeAddMagma` using `pure` instead of `of`."]
protected def recOnPure {C : FreeMagma α → Sort l} (x) (ih1 : ∀ x, C (pure x))
    (ih2 : ∀ x y, C x → C y → C (x * y)) : C x :=
  FreeMagma.recOnMul x ih1 ih2


@[to_additive (attr := simp)]
theorem map_pure (f : α → β) (x) : (f <$> pure x : FreeMagma β) = pure (f x) := rfl


@[to_additive (attr := simp)]
theorem map_mul' (f : α → β) (x y : FreeMagma α) : f <$> (x * y) = f <$> x * f <$> y := rfl


@[to_additive (attr := simp)]
theorem pure_bind (f : α → FreeMagma β) (x) : pure x >>= f = f x := rfl


@[to_additive (attr := simp)]
theorem mul_bind (f : α → FreeMagma β) (x y : FreeMagma α) : x * y >>= f = (x >>= f) * (y >>= f) :=
  rfl


@[to_additive (attr := simp)]
theorem pure_seq {α β : Type u} {f : α → β} {x : FreeMagma α} : pure f <*> x = f <$> x := rfl


@[to_additive (attr := simp)]
theorem mul_seq {α β : Type u} {f g : FreeMagma (α → β)} {x : FreeMagma α} :
    f * g <*> x = (f <*> x) * (g <*> x) := rfl


@[to_additive]
                                                        /-
                                                          α β : Type u
                                                          ⊢ ∀ {α β : Type u} (x : α) (y : FreeMagma β), Eq (Functor.mapConst x y) (Funct …
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                        /-
                                                          🎉 no goals
                                                        -/
    /-
      α β α✝ β✝ γ✝ : Type u
      x✝ : FreeMagma α✝
      f : α✝ → FreeMagma β✝
      g : β✝ → FreeMagma γ✝
      x y : FreeMagma α✝
      ih1 : Eq (Bind.bind (Bind.bind x f) g) (Bind.bind x fun x => Bind.bind (f x) g)
      ih2 : Eq (Bind.bind (Bind.bind y f) g) (Bind.bind y fun x => Bind.bind (f x) g)
      ⊢ Eq (Bind.bind (Bind.bind (HMul.hMul x y) f) g) (Bind.bind (HMul.hMul x y) fu …
    -/
                                                        /-
                                                          🎉 no goals
                                                        -/
    /-
      α β α✝ : Type u
      x✝ x y : FreeMagma α✝
      ih1 : Eq (Functor.map id x) x
      ih2 : Eq (Functor.map id y) y
      ⊢ Eq (Functor.map id (HMul.hMul x y)) (HMul.hMul x y)
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
instance instLawfulMonad : LawfulMonad FreeMagma.{u} := LawfulMonad.mk'
                                                        /-
                                                          🎉 no goals
                                                        -/
  (pure_bind := fun _ _ ↦ rfl)
  (bind_assoc := fun x f g ↦ FreeMagma.recOnPure x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by
    rw [mul_bind, mul_bind, mul_bind, ih1, ih2])
  (id_map := fun x ↦ FreeMagma.recOnPure x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by
    rw [map_mul', ih1, ih2])


/-- `FreeMagma` is traversable. -/
protected def FreeMagma.traverse {m : Type u → Type u} [Applicative m] {α β : Type u}
    (F : α → m β) : FreeMagma α → m (FreeMagma β)
  | FreeMagma.of x => FreeMagma.of <$> F x
  | mul x y => (· * ·) <$> x.traverse F <*> y.traverse F


/-- `FreeAddMagma` is traversable. -/
protected def FreeAddMagma.traverse {m : Type u → Type u} [Applicative m] {α β : Type u}
    (F : α → m β) : FreeAddMagma α → m (FreeAddMagma β)
  | FreeAddMagma.of x => FreeAddMagma.of <$> F x
  | x + y => (· + ·) <$> x.traverse F <*> y.traverse F


attribute [to_additive existing] FreeMagma.traverse


@[to_additive]
instance : Traversable FreeMagma := ⟨@FreeMagma.traverse⟩


@[to_additive (attr := simp)]
theorem traverse_pure (x) : traverse F (pure x : FreeMagma α) = pure <$> F x := rfl


@[to_additive (attr := simp)]
theorem traverse_pure' : traverse F ∘ pure = fun x ↦ (pure <$> F x : m (FreeMagma β)) := rfl


@[to_additive (attr := simp)]
theorem traverse_mul (x y : FreeMagma α) :
    traverse F (x * y) = (· * ·) <$> traverse F x <*> traverse F y := rfl


@[to_additive (attr := simp)]
theorem traverse_mul' :
    Function.comp (traverse F) ∘ (HMul.hMul : FreeMagma α → FreeMagma α → FreeMagma α) = fun x y ↦
      (· * ·) <$> traverse F x <*> traverse F y := rfl


@[to_additive (attr := simp)]
theorem traverse_eq (x) : FreeMagma.traverse F x = traverse F x := rfl

-- This is not a simp lemma because the left-hand side is not in simp normal form.

@[to_additive]
theorem mul_map_seq (x y : FreeMagma α) :
    ((· * ·) <$> x <*> y : Id (FreeMagma α)) = (x * y : FreeMagma α) := rfl


@[to_additive]
instance : LawfulTraversable FreeMagma.{u} :=
  { instLawfulMonad with
    id_traverse := fun x ↦
      FreeMagma.recOnPure x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by
        /-
          α β : Type u
          m : Type u → Type u
          inst✝ : Applicative m
          F : α → m β
          α✝ : Type u
          x✝ x y : FreeMagma α✝
          ih1 : Eq (Traversable.traverse Pure.pure x) x
          ih2 : Eq (Traversable.traverse Pure.pure y) y
          ⊢ Eq (Traversable.traverse Pure.pure (HMul.hMul x y)) (HMul.hMul x y)
        -/
        rw [traverse_mul, ih1, ih2, mul_map_seq]
        /-
          🎉 no goals
        -/
    comp_traverse := fun f g x ↦
      FreeMagma.recOnPure x
                    /-
                      α β : Type u
                      m : Type u → Type u
                      inst✝⁴ : Applicative m
                      F : α → m β
                      F✝ G✝ : Type u → Type u
                      inst✝³ : Applicative F✝
                      inst✝² : Applicative G✝
                      inst✝¹ : LawfulApplicative F✝
                      inst✝ : LawfulApplicative G✝
                      α✝ β✝ γ✝ : Type u
                      f : β✝ → F✝ γ✝
                      g : α✝ → G✝ β✝
                      x✝ : FreeMagma α✝
                      x : α✝
                      ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
                    -/
        (fun x ↦ by simp only [Function.comp_def, traverse_pure, traverse_pure', functor_norm])
                    /-
                      🎉 no goals
                    -/
        (fun x y ih1 ih2 ↦ by
          /-
            α β : Type u
            m : Type u → Type u
            inst✝⁴ : Applicative m
            F : α → m β
            F✝ G✝ : Type u → Type u
            inst✝³ : Applicative F✝
            inst✝² : Applicative G✝
            inst✝¹ : LawfulApplicative F✝
            inst✝ : LawfulApplicative G✝
            α✝ β✝ γ✝ : Type u
            f : β✝ → F✝ γ✝
            g : α✝ → G✝ β✝
            x✝ x y : FreeMagma α✝
            ih1 : Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp ( …
            ih2 : Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp ( …
            ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
          -/
          rw [traverse_mul, ih1, ih2, traverse_mul]
          simp [Functor.Comp.map_mk, Functor.map_map, Function.comp_def, Comp.seq_mk, seq_map_assoc,
            map_seq, traverse_mul])
    naturality := fun η α β f x ↦
      FreeMagma.recOnPure x
                    /-
                      α✝ β✝ : Type u
                      m : Type u → Type u
                      inst✝⁴ : Applicative m
                      F : α✝ → m β✝
                      F✝ G✝ : Type u → Type u
                      inst✝³ : Applicative F✝
                      inst✝² : Applicative G✝
                      inst✝¹ : LawfulApplicative F✝
                      inst✝ : LawfulApplicative G✝
                      η : ApplicativeTransformation F✝ G✝
                      α β : Type u
                      f : α → F✝ β
                      x✝ : FreeMagma α
                      x : α
                      ⊢ Eq ((fun {α} => η.app α) (Traversable.traverse f (Pure.pure x))) (Traversabl …
                    -/
        (fun x ↦ by simp only [traverse_pure, functor_norm, Function.comp_apply])
                    /-
                      🎉 no goals
                    -/
                              /-
                                α✝ β✝ : Type u
                                m : Type u → Type u
                                inst✝⁴ : Applicative m
                                F : α✝ → m β✝
                                F✝ G✝ : Type u → Type u
                                inst✝³ : Applicative F✝
                                inst✝² : Applicative G✝
                                inst✝¹ : LawfulApplicative F✝
                                inst✝ : LawfulApplicative G✝
                                η : ApplicativeTransformation F✝ G✝
                                α β : Type u
                                f : α → F✝ β
                                x✝ x y : FreeMagma α
                                ih1 : Eq ((fun {α} => η.app α) (Traversable.traverse f x)) (Traversable.traver …
                                ih2 : Eq ((fun {α} => η.app α) (Traversable.traverse f y)) (Traversable.traver …
                                ⊢ Eq ((fun {α} => η.app α) (Traversable.traverse f (HMul.hMul x y))) (Traversa …
                              -/
        /-
          α β : Type u
          m : Type u → Type u
          inst✝ : Applicative m
          F : α → m β
          α✝ β✝ : Type u
          f : α✝ → β✝
          x✝ x y : FreeMagma α✝
          ih1 : Eq (Traversable.traverse (Function.comp Pure.pure f) x) (id.mk (Functor. …
          ih2 : Eq (Traversable.traverse (Function.comp Pure.pure f) y) (id.mk (Functor. …
          ⊢ Eq (Traversable.traverse (Function.comp Pure.pure f) (HMul.hMul x y)) (id.mk …
        -/
        (fun x y ih1 ih2 ↦ by simp only [traverse_mul, functor_norm, ih1, ih2])
                                                            /-
                                                              🎉 no goals
                                                            -/
                              /-
                                🎉 no goals
                              -/
    traverse_eq_map_id := fun f x ↦
      FreeMagma.recOnPure x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by
        rw [traverse_mul, ih1, ih2, map_mul', mul_map_seq]; rfl }


/-- Representation of an element of a free magma. -/
protected def FreeMagma.repr {α : Type u} [Repr α] : FreeMagma α → Lean.Format
  | FreeMagma.of x => repr x
  | mul x y => "( " ++ x.repr ++ " * " ++ y.repr ++ " )"


/-- Representation of an element of a free additive magma. -/
protected def FreeAddMagma.repr {α : Type u} [Repr α] : FreeAddMagma α → Lean.Format
  | FreeAddMagma.of x => repr x
  | x + y => "( " ++ x.repr ++ " + " ++ y.repr ++ " )"


attribute [to_additive existing] FreeMagma.repr


@[to_additive]
instance {α : Type u} [Repr α] : Repr (FreeMagma α) := ⟨fun o _ => FreeMagma.repr o⟩


/-- Length of an element of a free magma. -/
def FreeMagma.length {α : Type u} : FreeMagma α → ℕ
  | FreeMagma.of _x => 1
  | mul x y => x.length + y.length


/-- Length of an element of a free additive magma. -/
def FreeAddMagma.length {α : Type u} : FreeAddMagma α → ℕ
  | FreeAddMagma.of _x => 1
  | x + y => x.length + y.length


attribute [to_additive existing (attr := simp)] FreeMagma.length


/-- The length of an element of a free magma is positive. -/
@[to_additive "The length of an element of a free additive magma is positive."]
lemma FreeMagma.length_pos {α : Type u} (x : FreeMagma α) : 0 < x.length :=
  match x with
  | FreeMagma.of _ => Nat.succ_pos 0
  | mul y z => Nat.add_pos_left (length_pos y) z.length


/-- Associativity relations for an additive magma. -/
inductive AddMagma.AssocRel (α : Type u) [Add α] : α → α → Prop
  | intro : ∀ x y z, AddMagma.AssocRel α (x + y + z) (x + (y + z))
  | left : ∀ w x y z, AddMagma.AssocRel α (w + (x + y + z)) (w + (x + (y + z)))


/-- Associativity relations for a magma. -/
@[to_additive AddMagma.AssocRel "Associativity relations for an additive magma."]
inductive Magma.AssocRel (α : Type u) [Mul α] : α → α → Prop
  | intro : ∀ x y z, Magma.AssocRel α (x * y * z) (x * (y * z))
  | left : ∀ w x y z, Magma.AssocRel α (w * (x * y * z)) (w * (x * (y * z)))


/-- Semigroup quotient of a magma. -/
@[to_additive AddMagma.FreeAddSemigroup "Additive semigroup quotient of an additive magma."]
def AssocQuotient (α : Type u) [Mul α] : Type u :=
  Quot <| AssocRel α


@[to_additive]
theorem quot_mk_assoc (x y z : α) : Quot.mk (AssocRel α) (x * y * z) = Quot.mk _ (x * (y * z)) :=
  Quot.sound (AssocRel.intro _ _ _)


@[to_additive]
theorem quot_mk_assoc_left (x y z w : α) :
    Quot.mk (AssocRel α) (x * (y * z * w)) = Quot.mk _ (x * (y * (z * w))) :=
  Quot.sound (AssocRel.left _ _ _ _)


@[to_additive]
instance : Semigroup (AssocQuotient α) where
  mul x y := by
    /-
      α : Type u
      inst✝ : Mul α
      x y : Magma.AssocQuotient α
      ⊢ Magma.AssocQuotient α
    -/
    refine Quot.liftOn₂ x y (fun x y ↦ Quot.mk _ (x * y)) ?_ ?_
      /-
        case refine_1
        α : Type u
        inst✝ : Mul α
        x y : Magma.AssocQuotient α
        ⊢ ∀ (a b₁ b₂ : α), Magma.AssocRel α b₁ b₂ → Eq ((fun x y => Quot.mk (Magma.Ass …
      -/
    · rintro a b₁ b₂ (⟨c, d, e⟩ | ⟨c, d, e, f⟩) <;> simp only
        /-
          case refine_1.intro
          α : Type u
          inst✝ : Mul α
          x y : Magma.AssocQuotient α
          a c d e : α
          ⊢ Eq (Quot.mk (Magma.AssocRel α) (HMul.hMul a (HMul.hMul (HMul.hMul c d) e)))  …
        -/
      · exact quot_mk_assoc_left _ _ _ _
        /-
          🎉 no goals
        -/
        /-
          case refine_1.left
          α : Type u
          inst✝ : Mul α
          x y : Magma.AssocQuotient α
          a c d e f : α
          ⊢ Eq (Quot.mk (Magma.AssocRel α) (HMul.hMul a (HMul.hMul c (HMul.hMul (HMul.hM …
        -/
      · rw [← quot_mk_assoc, quot_mk_assoc_left, quot_mk_assoc]
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        α : Type u
        inst✝ : Mul α
        x y : Magma.AssocQuotient α
        ⊢ ∀ (a₁ a₂ b : α), Magma.AssocRel α a₁ a₂ → Eq ((fun x y => Quot.mk (Magma.Ass …
      -/
    · rintro a₁ a₂ b (⟨c, d, e⟩ | ⟨c, d, e, f⟩) <;> simp only
        /-
          case refine_2.intro
          α : Type u
          inst✝ : Mul α
          x y : Magma.AssocQuotient α
          b c d e : α
          ⊢ Eq (Quot.mk (Magma.AssocRel α) (HMul.hMul (HMul.hMul (HMul.hMul c d) e) b))  …
        -/
      · simp only [quot_mk_assoc, quot_mk_assoc_left]
        /-
          🎉 no goals
        -/
      · rw [quot_mk_assoc, quot_mk_assoc, quot_mk_assoc_left, quot_mk_assoc_left,
          quot_mk_assoc_left, ← quot_mk_assoc c d, ← quot_mk_assoc c d, quot_mk_assoc_left]
  mul_assoc x y z :=
    Quot.induction_on₃ x y z fun a b c ↦ quot_mk_assoc a b c


/-- Embedding from magma to its free semigroup. -/
@[to_additive "Embedding from additive magma to its free additive semigroup."]
def of : α →ₙ* AssocQuotient α where toFun := Quot.mk _; map_mul' _x _y := rfl


@[to_additive]
instance [Inhabited α] : Inhabited (AssocQuotient α) := ⟨of default⟩


@[to_additive (attr := elab_as_elim, induction_eliminator)]
protected theorem induction_on {C : AssocQuotient α → Prop} (x : AssocQuotient α)
    (ih : ∀ x, C (of x)) : C x := Quot.induction_on x ih


@[to_additive (attr := ext 1100)]
theorem hom_ext {f g : AssocQuotient α →ₙ* β} (h : f.comp of = g.comp of) : f = g :=
  (DFunLike.ext _ _) fun x => AssocQuotient.induction_on x <| DFunLike.congr_fun h


/-- Lifts a magma homomorphism `α → β` to a semigroup homomorphism `Magma.AssocQuotient α → β`
given a semigroup `β`. -/
@[to_additive (attr := simps symm_apply) "Lifts an additive magma homomorphism `α → β` to an
additive semigroup homomorphism `AddMagma.AssocQuotient α → β` given an additive semigroup `β`."]
def lift : (α →ₙ* β) ≃ (AssocQuotient α →ₙ* β) where
  toFun f :=
  { toFun := fun x ↦
                            /-
                              α : Type u
                              inst✝¹ : Mul α
                              β : Type v
                              inst✝ : Semigroup β
                              f✝ f : MulHom α β
                              x : Magma.AssocQuotient α
                              ⊢ ∀ (a b : α), Magma.AssocRel α a b → Eq (f a) (f b)
                            -/
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
      Quot.liftOn x f <| by rintro a b (⟨c, d, e⟩ | ⟨c, d, e, f⟩) <;> simp only [map_mul, mul_assoc]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    map_mul' := fun x y ↦ Quot.induction_on₂ x y (map_mul f) }
  invFun f := f.comp of
  left_inv _ := (DFunLike.ext _ _) fun _ ↦ rfl
  right_inv _ := hom_ext <| (DFunLike.ext _ _) fun _ ↦ rfl


@[to_additive (attr := simp)]
theorem lift_of (x : α) : lift f (of x) = f x := rfl


@[to_additive (attr := simp)]
theorem lift_comp_of : (lift f).comp of = f := lift.symm_apply_apply f


@[to_additive (attr := simp)]
theorem lift_comp_of' (f : AssocQuotient α →ₙ* β) : lift (f.comp of) = f := lift.apply_symm_apply f


/-- From a magma homomorphism `α →ₙ* β` to a semigroup homomorphism
`Magma.AssocQuotient α →ₙ* Magma.AssocQuotient β`. -/
@[to_additive "From an additive magma homomorphism `α → β` to an additive semigroup homomorphism
`AddMagma.AssocQuotient α → AddMagma.AssocQuotient β`."]
def map : AssocQuotient α →ₙ* AssocQuotient β := lift (of.comp f)


/-- Free additive semigroup over a given alphabet. -/
structure FreeAddSemigroup (α : Type u) where
  /-- The head of the element -/
  head : α
  /-- The tail of the element -/
  tail : List α

compile_inductive% FreeAddSemigroup


/-- Free semigroup over a given alphabet. -/
@[to_additive (attr := ext)]
structure FreeSemigroup (α : Type u) where
  /-- The head of the element -/
  head : α
  /-- The tail of the element -/
  tail : List α

compile_inductive% FreeSemigroup


@[to_additive]
instance : Semigroup (FreeSemigroup α) where
  mul L1 L2 := ⟨L1.1, L1.2 ++ L2.1 :: L2.2⟩
  mul_assoc _L1 _L2 _L3 := FreeSemigroup.ext rfl <| List.append_assoc _ _ _


@[to_additive (attr := simp)]
theorem head_mul (x y : FreeSemigroup α) : (x * y).1 = x.1 := rfl


@[to_additive (attr := simp)]
theorem tail_mul (x y : FreeSemigroup α) : (x * y).2 = x.2 ++ y.1 :: y.2 := rfl


@[to_additive (attr := simp)]
theorem mk_mul_mk (x y : α) (L1 L2 : List α) : mk x L1 * mk y L2 = mk x (L1 ++ y :: L2) := rfl


/-- The embedding `α → FreeSemigroup α`. -/
@[to_additive (attr := simps) "The embedding `α → FreeAddSemigroup α`."]
def of (x : α) : FreeSemigroup α := ⟨x, []⟩


/-- Length of an element of free semigroup. -/
@[to_additive "Length of an element of free additive semigroup"]
def length (x : FreeSemigroup α) : ℕ := x.tail.length + 1


@[to_additive (attr := simp)]
theorem length_mul (x y : FreeSemigroup α) : (x * y).length = x.length + y.length := by
  /-
    α : Type u
    x y : FreeSemigroup α
    ⊢ Eq (HMul.hMul x y).length (HAdd.hAdd x.length y.length)
  -/
  simp [length, Nat.add_right_comm, List.length, List.length_append]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem length_of (x : α) : (of x).length = 1 := rfl


@[to_additive]
instance [Inhabited α] : Inhabited (FreeSemigroup α) := ⟨of default⟩


/-- Recursor for free semigroup using `of` and `*`. -/
@[to_additive (attr := elab_as_elim, induction_eliminator)
  "Recursor for free additive semigroup using `of` and `+`."]
protected def recOnMul {C : FreeSemigroup α → Sort l} (x) (ih1 : ∀ x, C (of x))
    (ih2 : ∀ x y, C (of x) → C y → C (of x * y)) : C x :=
      FreeSemigroup.recOn x fun f s ↦
      List.recOn s ih1 (fun hd tl ih f ↦ ih2 f ⟨hd, tl⟩ (ih1 f) (ih hd)) f


@[to_additive (attr := ext 1100)]
theorem hom_ext {β : Type v} [Mul β] {f g : FreeSemigroup α →ₙ* β} (h : f ∘ of = g ∘ of) : f = g :=
  (DFunLike.ext _ _) fun x ↦
                                                              /-
                                                                α : Type u
                                                                β : Type v
                                                                inst✝ : Mul β
                                                                f g : MulHom (FreeSemigroup α) β
                                                                h : Eq (Function.comp (⇑f) FreeSemigroup.of) (Function.comp (⇑g) FreeSemigroup …
                                                                x✝ : FreeSemigroup α
                                                                x : α
                                                                y : FreeSemigroup α
                                                                hx : Eq (f (FreeSemigroup.of x)) (g (FreeSemigroup.of x))
                                                                hy : Eq (f y) (g y)
                                                                ⊢ Eq (f (HMul.hMul (FreeSemigroup.of x) y)) (g (HMul.hMul (FreeSemigroup.of x) …
                                                              -/
    FreeSemigroup.recOnMul x (congr_fun h) fun x y hx hy ↦ by simp only [map_mul, *]
                                                              /-
                                                                🎉 no goals
                                                              -/


/-- Lifts a function `α → β` to a semigroup homomorphism `FreeSemigroup α → β` given
a semigroup `β`. -/
@[to_additive (attr := simps symm_apply) "Lifts a function `α → β` to an additive semigroup
homomorphism `FreeAddSemigroup α → β` given an additive semigroup `β`."]
def lift : (α → β) ≃ (FreeSemigroup α →ₙ* β) where
  toFun f :=
    { toFun := fun x ↦ x.2.foldl (fun a b ↦ a * f b) (f x.1)
      map_mul' := fun x y ↦ by
        simp [head_mul, tail_mul, ← List.foldl_map, List.foldl_append, List.foldl_cons,
          List.foldl_assoc] }
  invFun f := f ∘ of
  left_inv _ := rfl
  right_inv _ := hom_ext rfl


@[to_additive (attr := simp)]
theorem lift_comp_of' (f : FreeSemigroup α →ₙ* β) : lift (f ∘ of) = f := hom_ext rfl


@[to_additive]
                                                                     /-
                                                                       α : Type u
                                                                       β : Type v
                                                                       inst✝ : Semigroup β
                                                                       f : α → β
                                                                       x : α
                                                                       y : FreeSemigroup α
                                                                       ⊢ Eq ((FreeSemigroup.lift f) (HMul.hMul (FreeSemigroup.of x) y)) (HMul.hMul (f …
                                                                     -/
theorem lift_of_mul (x y) : lift f (of x * y) = f x * lift f y := by rw [map_mul, lift_of]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- The unique semigroup homomorphism that sends `of x` to `of (f x)`. -/
@[to_additive "The unique additive semigroup homomorphism that sends `of x` to `of (f x)`."]
def map : FreeSemigroup α →ₙ* FreeSemigroup β :=
  lift <| of ∘ f


@[to_additive (attr := simp)]
theorem length_map (x) : (map f x).length = x.length :=
                                                             /-
                                                               α : Type u
                                                               β : Type v
                                                               f : α → β
                                                               x✝ : FreeSemigroup α
                                                               x : α
                                                               y : FreeSemigroup α
                                                               hx : Eq ((FreeSemigroup.map f) (FreeSemigroup.of x)).length (FreeSemigroup.of  …
                                                               hy : Eq ((FreeSemigroup.map f) y).length y.length
                                                               ⊢ Eq ((FreeSemigroup.map f) (HMul.hMul (FreeSemigroup.of x) y)).length (HMul.h …
                                                             -/
  FreeSemigroup.recOnMul x (fun _ ↦ rfl) (fun x y hx hy ↦ by simp only [map_mul, length_mul, *])
                                                             /-
                                                               🎉 no goals
                                                             -/


@[to_additive]
instance : Monad FreeSemigroup where
  pure := of
  bind x f := lift f x


/-- Recursor that uses `pure` instead of `of`. -/
@[to_additive (attr := elab_as_elim) "Recursor that uses `pure` instead of `of`."]
def recOnPure {C : FreeSemigroup α → Sort l} (x) (ih1 : ∀ x, C (pure x))
    (ih2 : ∀ x y, C (pure x) → C y → C (pure x * y)) : C x :=
  FreeSemigroup.recOnMul x ih1 ih2


@[to_additive (attr := simp)]
theorem map_pure (f : α → β) (x) : (f <$> pure x : FreeSemigroup β) = pure (f x) := rfl


@[to_additive (attr := simp)]
theorem map_mul' (f : α → β) (x y : FreeSemigroup α) : f <$> (x * y) = f <$> x * f <$> y :=
  map_mul (map f) _ _


@[to_additive (attr := simp)]
theorem pure_bind (f : α → FreeSemigroup β) (x) : pure x >>= f = f x := rfl


@[to_additive (attr := simp)]
theorem mul_bind (f : α → FreeSemigroup β) (x y : FreeSemigroup α) :
    x * y >>= f = (x >>= f) * (y >>= f) := map_mul (lift f) _ _


@[to_additive (attr := simp)]
theorem pure_seq {f : α → β} {x : FreeSemigroup α} : pure f <*> x = f <$> x := rfl


@[to_additive (attr := simp)]
theorem mul_seq {f g : FreeSemigroup (α → β)} {x : FreeSemigroup α} :
    f * g <*> x = (f <*> x) * (g <*> x) := mul_bind _ _ _


@[to_additive]
                                                            /-
                                                              α β : Type u
                                                              ⊢ ∀ {α β : Type u} (x : α) (y : FreeSemigroup β), Eq (Functor.mapConst x y) (F …
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                   /-
                                                     α β α✝ β✝ γ✝ : Type u
                                                     x✝ : FreeSemigroup α✝
                                                     g : α✝ → FreeSemigroup β✝
                                                     f : β✝ → FreeSemigroup γ✝
                                                     x : α✝
                                                     y : FreeSemigroup α✝
                                                     ih1 : Eq (Bind.bind (Bind.bind (Pure.pure x) g) f) (Bind.bind (Pure.pure x) fu …
                                                     ih2 : Eq (Bind.bind (Bind.bind y g) f) (Bind.bind y fun x => Bind.bind (g x) f)
                                                     ⊢ Eq (Bind.bind (Bind.bind (HMul.hMul (Pure.pure x) y) g) f) (Bind.bind (HMul. …
                                                   -/
                                                                    /-
                                                                      α β α✝ : Type u
                                                                      x✝ : FreeSemigroup α✝
                                                                      x : α✝
                                                                      y : FreeSemigroup α✝
                                                                      ih1 : Eq (Functor.map id (Pure.pure x)) (Pure.pure x)
                                                                      ih2 : Eq (Functor.map id y) y
                                                                      ⊢ Eq (Functor.map id (HMul.hMul (Pure.pure x) y)) (HMul.hMul (Pure.pure x) y)
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
instance instLawfulMonad : LawfulMonad FreeSemigroup.{u} := LawfulMonad.mk'
                                                            /-
                                                              🎉 no goals
                                                            -/
  (pure_bind := fun _ _ ↦ rfl)
  (bind_assoc := fun x g f ↦
    recOnPure x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by rw [mul_bind, mul_bind, mul_bind, ih1, ih2])
  (id_map := fun x ↦ recOnPure x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by rw [map_mul', ih1, ih2])


/-- `FreeSemigroup` is traversable. -/
@[to_additive "`FreeAddSemigroup` is traversable."]
protected def traverse {m : Type u → Type u} [Applicative m] {α β : Type u}
    (F : α → m β) (x : FreeSemigroup α) : m (FreeSemigroup β) :=
  recOnPure x (fun x ↦ pure <$> F x) fun _x _y ihx ihy ↦ (· * ·) <$> ihx <*> ihy


@[to_additive]
instance : Traversable FreeSemigroup := ⟨@FreeSemigroup.traverse⟩


@[to_additive (attr := simp)]
theorem traverse_pure (x) : traverse F (pure x : FreeSemigroup α) = pure <$> F x := rfl


@[to_additive (attr := simp)]
theorem traverse_pure' : traverse F ∘ pure = fun x ↦ (pure <$> F x : m (FreeSemigroup β)) := rfl


@[to_additive (attr := simp)]
theorem traverse_mul (x y : FreeSemigroup α) :
    traverse F (x * y) = (· * ·) <$> traverse F x <*> traverse F y :=
  let ⟨x, L1⟩ := x
  let ⟨y, L2⟩ := y
  List.recOn L1 (fun _ ↦ rfl)
    (fun hd tl ih x ↦ show
        (· * ·) <$> pure <$> F x <*> traverse F (mk hd tl * mk y L2) =
          (· * ·) <$> ((· * ·) <$> pure <$> F x <*> traverse F (mk hd tl)) <*> traverse F (mk y L2)
           /-
             α β : Type u
             m : Type u → Type u
             inst✝¹ : Applicative m
             F : α → m β
             inst✝ : LawfulApplicative m
             x✝¹ y✝ : FreeSemigroup α
             x✝ : α
             L1 : List α
             y : α
             L2 : List α
             hd : α
             tl : List α
             ih : ∀ (x : α), Eq (Traversable.traverse F (HMul.hMul { head := x, tail := tl  …
             x : α
             ⊢ Eq (Seq.seq (Functor.map (fun x1 x2 => HMul.hMul x1 x2) (Functor.map Pure.pu …
           -/
        by rw [ih]; simp only [Function.comp_def, (mul_assoc _ _ _).symm, functor_norm])
                    /-
                      🎉 no goals
                    -/
    x


@[to_additive (attr := simp)]
theorem traverse_mul' :
    Function.comp (traverse F) ∘ (HMul.hMul : FreeSemigroup α → FreeSemigroup α → FreeSemigroup α) =
      fun x y ↦ (· * ·) <$> traverse F x <*> traverse F y :=
  funext fun x ↦ funext fun y ↦ traverse_mul F x y


@[to_additive (attr := simp)]
theorem traverse_eq (x) : FreeSemigroup.traverse F x = traverse F x := rfl

-- This is not a simp lemma because the left-hand side is not in simp normal form.

@[to_additive]
theorem mul_map_seq (x y : FreeSemigroup α) :
    ((· * ·) <$> x <*> y : Id (FreeSemigroup α)) = (x * y : FreeSemigroup α) := rfl


@[to_additive]
instance : LawfulTraversable FreeSemigroup.{u} :=
  { instLawfulMonad with
    id_traverse := fun x ↦
      FreeSemigroup.recOnMul x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by
        /-
          α β : Type u
          m : Type u → Type u
          inst✝ : Applicative m
          F : α → m β
          α✝ : Type u
          x✝ : FreeSemigroup α✝
          x : α✝
          y : FreeSemigroup α✝
          ih1 : Eq (Traversable.traverse Pure.pure (FreeSemigroup.of x)) (FreeSemigroup. …
          ih2 : Eq (Traversable.traverse Pure.pure y) y
          ⊢ Eq (Traversable.traverse Pure.pure (HMul.hMul (FreeSemigroup.of x) y)) (HMul …
        -/
        rw [traverse_mul, ih1, ih2, mul_map_seq]
        /-
          🎉 no goals
        -/
    comp_traverse := fun f g x ↦
                              /-
                                α β : Type u
                                m : Type u → Type u
                                inst✝⁴ : Applicative m
                                F : α → m β
                                F✝ G✝ : Type u → Type u
                                inst✝³ : Applicative F✝
                                inst✝² : Applicative G✝
                                inst✝¹ : LawfulApplicative F✝
                                inst✝ : LawfulApplicative G✝
                                α✝ β✝ γ✝ : Type u
                                f : β✝ → F✝ γ✝
                                g : α✝ → G✝ β✝
                                x✝ : FreeSemigroup α✝
                                x : α✝
                                ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
                              -/
      recOnPure x (fun x ↦ by simp only [traverse_pure, functor_norm, Function.comp_def])
                              /-
                                🎉 no goals
                              -/
        fun x y ih1 ih2 ↦ by
          /-
            α β : Type u
            m : Type u → Type u
            inst✝⁴ : Applicative m
            F : α → m β
            F✝ G✝ : Type u → Type u
            inst✝³ : Applicative F✝
            inst✝² : Applicative G✝
            inst✝¹ : LawfulApplicative F✝
            inst✝ : LawfulApplicative G✝
            α✝ β✝ γ✝ : Type u
            f : β✝ → F✝ γ✝
            g : α✝ → G✝ β✝
            x✝ : FreeSemigroup α✝
            x : α✝
            y : FreeSemigroup α✝
            ih1 : Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp ( …
            ih2 : Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp ( …
            ⊢ Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp (Func …
          -/
          rw [traverse_mul, ih1, ih2, traverse_mul, Functor.Comp.map_mk]
          /-
            α β : Type u
            m : Type u → Type u
            inst✝⁴ : Applicative m
            F : α → m β
            F✝ G✝ : Type u → Type u
            inst✝³ : Applicative F✝
            inst✝² : Applicative G✝
            inst✝¹ : LawfulApplicative F✝
            inst✝ : LawfulApplicative G✝
            α✝ β✝ γ✝ : Type u
            f : β✝ → F✝ γ✝
            g : α✝ → G✝ β✝
            x✝ : FreeSemigroup α✝
            x : α✝
            y : FreeSemigroup α✝
            ih1 : Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp ( …
            ih2 : Eq (Traversable.traverse (Function.comp Functor.Comp.mk (Function.comp ( …
            ⊢ Eq (Seq.seq (Functor.Comp.mk (Functor.map (fun x => Functor.map (fun x1 x2 = …
          -/
          simp only [Function.comp_def, functor_norm, traverse_mul]
          /-
            🎉 no goals
          -/
    naturality := fun η α β f x ↦
                              /-
                                α✝ β✝ : Type u
                                m : Type u → Type u
                                inst✝⁴ : Applicative m
                                F : α✝ → m β✝
                                F✝ G✝ : Type u → Type u
                                inst✝³ : Applicative F✝
                                inst✝² : Applicative G✝
                                inst✝¹ : LawfulApplicative F✝
                                inst✝ : LawfulApplicative G✝
                                η : ApplicativeTransformation F✝ G✝
                                α β : Type u
                                f : α → F✝ β
                                x✝ : FreeSemigroup α
                                x : α
                                ⊢ Eq ((fun {α} => η.app α) (Traversable.traverse f (Pure.pure x))) (Traversabl …
                              -/
      recOnPure x (fun x ↦ by simp only [traverse_pure, functor_norm, Function.comp])
                              /-
                                🎉 no goals
                              -/
                                /-
                                  α✝ β✝ : Type u
                                  m : Type u → Type u
                                  inst✝⁴ : Applicative m
                                  F : α✝ → m β✝
                                  F✝ G✝ : Type u → Type u
                                  inst✝³ : Applicative F✝
                                  inst✝² : Applicative G✝
                                  inst✝¹ : LawfulApplicative F✝
                                  inst✝ : LawfulApplicative G✝
                                  η : ApplicativeTransformation F✝ G✝
                                  α β : Type u
                                  f : α → F✝ β
                                  x✝ : FreeSemigroup α
                                  x : α
                                  y : FreeSemigroup α
                                  ih1 : Eq ((fun {α} => η.app α) (Traversable.traverse f (Pure.pure x))) (Traver …
                                  ih2 : Eq ((fun {α} => η.app α) (Traversable.traverse f y)) (Traversable.traver …
                                  ⊢ Eq ((fun {α} => η.app α) (Traversable.traverse f (HMul.hMul (Pure.pure x) y) …
                                -/
        /-
          α β : Type u
          m : Type u → Type u
          inst✝ : Applicative m
          F : α → m β
          α✝ β✝ : Type u
          f : α✝ → β✝
          x✝ : FreeSemigroup α✝
          x : α✝
          y : FreeSemigroup α✝
          ih1 : Eq (Traversable.traverse (Function.comp Pure.pure f) (FreeSemigroup.of x …
          ih2 : Eq (Traversable.traverse (Function.comp Pure.pure f) y) (id.mk (Functor. …
          ⊢ Eq (Traversable.traverse (Function.comp Pure.pure f) (HMul.hMul (FreeSemigro …
        -/
          (fun x y ih1 ih2 ↦ by simp only [traverse_mul, functor_norm, ih1, ih2])
                                                            /-
                                                              🎉 no goals
                                                            -/
                                /-
                                  🎉 no goals
                                -/
    traverse_eq_map_id := fun f x ↦
      FreeSemigroup.recOnMul x (fun _ ↦ rfl) fun x y ih1 ih2 ↦ by
        rw [traverse_mul, ih1, ih2, map_mul', mul_map_seq]; rfl }


@[to_additive]
instance [DecidableEq α] : DecidableEq (FreeSemigroup α) :=
  fun _ _ ↦ decidable_of_iff' _ FreeSemigroup.ext_iff


/-- The canonical multiplicative morphism from `FreeMagma α` to `FreeSemigroup α`. -/
@[to_additive "The canonical additive morphism from `FreeAddMagma α` to `FreeAddSemigroup α`."]
def toFreeSemigroup : FreeMagma α →ₙ* FreeSemigroup α := FreeMagma.lift FreeSemigroup.of


@[to_additive (attr := simp)]
theorem toFreeSemigroup_of (x : α) : toFreeSemigroup (of x) = FreeSemigroup.of x := rfl


@[to_additive (attr := simp)]
theorem toFreeSemigroup_comp_of : @toFreeSemigroup α ∘ of = FreeSemigroup.of := rfl


@[to_additive]
theorem toFreeSemigroup_comp_map (f : α → β) :
                                                                                    /-
                                                                                      α : Type u
                                                                                      β : Type v
                                                                                      f : α → β
                                                                                      ⊢ Eq (FreeMagma.toFreeSemigroup.comp (FreeMagma.map f)) ((FreeSemigroup.map f) …
                                                                                    -/
    toFreeSemigroup.comp (map f) = (FreeSemigroup.map f).comp toFreeSemigroup := by ext1; rfl
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[to_additive]
theorem toFreeSemigroup_map (f : α → β) (x : FreeMagma α) :
    toFreeSemigroup (map f x) = FreeSemigroup.map f (toFreeSemigroup x) :=
  DFunLike.congr_fun (toFreeSemigroup_comp_map f) x


@[to_additive (attr := simp)]
theorem length_toFreeSemigroup (x : FreeMagma α) : (toFreeSemigroup x).length = x.length :=
  FreeMagma.recOnMul x (fun _ ↦ rfl) fun x y hx hy ↦ by
    /-
      α : Type u
      x✝ x y : FreeMagma α
      hx : Eq (FreeMagma.toFreeSemigroup x).length x.length
      hy : Eq (FreeMagma.toFreeSemigroup y).length y.length
      ⊢ Eq (FreeMagma.toFreeSemigroup (HMul.hMul x y)).length (HMul.hMul x y).length
    -/
    rw [map_mul, FreeSemigroup.length_mul, hx, hy]; rfl
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- Isomorphism between `Magma.AssocQuotient (FreeMagma α)` and `FreeSemigroup α`. -/
@[to_additive "Isomorphism between `AddMagma.AssocQuotient (FreeAddMagma α)` and
`FreeAddSemigroup α`."]
def FreeMagmaAssocQuotientEquiv (α : Type u) :
    Magma.AssocQuotient (FreeMagma α) ≃* FreeSemigroup α :=
      (Magma.AssocQuotient.lift FreeMagma.toFreeSemigroup).toMulEquiv
      (FreeSemigroup.lift (Magma.AssocQuotient.of ∘ FreeMagma.of))
          /-
            α : Type u
            ⊢ Eq ((FreeSemigroup.lift (Function.comp (⇑Magma.AssocQuotient.of) FreeMagma.o …
          -/
      (by ext; rfl)
               /-
                 🎉 no goals
               -/
          /-
            α : Type u
            ⊢ Eq ((Magma.AssocQuotient.lift FreeMagma.toFreeSemigroup).comp (FreeSemigroup …
          -/
      (by ext1; rfl)
                /-
                  🎉 no goals
                -/

