theorem Functor.map_id : (id <$> ·) = (id : F α → F α) := funext id_map


theorem Functor.map_comp_map (f : α → β) (g : β → γ) :
    ((g <$> ·) ∘ (f <$> ·) : F α → F γ) = ((g ∘ f) <$> ·) :=
  funext fun _ => (comp_map _ _ _).symm
  -- Porting note: was `apply funext <;> intro <;> rw [comp_map]` but `rw` failed?


theorem Functor.ext {F} :
    ∀ {F1 : Functor F} {F2 : Functor F} [@LawfulFunctor F F1] [@LawfulFunctor F F2],
    (∀ (α β) (f : α → β) (x : F α), @Functor.map _ F1 _ _ f x = @Functor.map _ F2 _ _ f x) →
    F1 = F2
  | ⟨m, mc⟩, ⟨m', mc'⟩, H1, H2, H => by
    /-
      F : Type u_1 → Type u_2
      m : {α β : Type u_1} → (α → β) → F α → F β
      mc : {α β : Type u_1} → α → F β → F α
      m' : {α β : Type u_1} → (α → β) → F α → F β
      mc' : {α β : Type u_1} → α → F β → F α
      H1 : LawfulFunctor F
      H2 : LawfulFunctor F
      H : ∀ (α β : Type u_1) (f : α → β) (x : F α), Eq (Functor.map f x) (Functor.ma …
      ⊢ Eq { map := m, mapConst := mc } { map := m', mapConst := mc' }
    -/
    cases show @m = @m' by funext α β f x; apply H
    /-
      case refl
      F : Type u_1 → Type u_2
      m : {α β : Type u_1} → (α → β) → F α → F β
      mc mc' : {α β : Type u_1} → α → F β → F α
      H1 : LawfulFunctor F
      H2 : LawfulFunctor F
      H : ∀ (α β : Type u_1) (f : α → β) (x : F α), Eq (Functor.map f x) (Functor.ma …
      ⊢ Eq { map := m, mapConst := mc } { map := m, mapConst := mc' }
    -/
    congr
    /-
      case refl.e_mapConst
      F : Type u_1 → Type u_2
      m : {α β : Type u_1} → (α → β) → F α → F β
      mc mc' : {α β : Type u_1} → α → F β → F α
      H1 : LawfulFunctor F
      H2 : LawfulFunctor F
      H : ∀ (α β : Type u_1) (f : α → β) (x : F α), Eq (Functor.map f x) (Functor.ma …
      ⊢ Eq mc mc'
    -/
    funext α β
    /-
      case refl.e_mapConst.h.h
      F : Type u_1 → Type u_2
      m : {α β : Type u_1} → (α → β) → F α → F β
      mc mc' : {α β : Type u_1} → α → F β → F α
      H1 : LawfulFunctor F
      H2 : LawfulFunctor F
      H : ∀ (α β : Type u_1) (f : α → β) (x : F α), Eq (Functor.map f x) (Functor.ma …
      α β : Type u_1
      ⊢ Eq mc mc'
    -/
    have E1 := @map_const _ ⟨@m, @mc⟩ H1
    /-
      case refl.e_mapConst.h.h
      F : Type u_1 → Type u_2
      m : {α β : Type u_1} → (α → β) → F α → F β
      mc mc' : {α β : Type u_1} → α → F β → F α
      H1 : LawfulFunctor F
      H2 : LawfulFunctor F
      H : ∀ (α β : Type u_1) (f : α → β) (x : F α), Eq (Functor.map f x) (Functor.ma …
      α β : Type u_1
      E1 : ∀ {α β : Type u_1}, Eq Functor.mapConst (Function.comp Functor.map (Funct …
      ⊢ Eq mc mc'
    -/
    have E2 := @map_const _ ⟨@m, @mc'⟩ H2
    /-
      case refl.e_mapConst.h.h
      F : Type u_1 → Type u_2
      m : {α β : Type u_1} → (α → β) → F α → F β
      mc mc' : {α β : Type u_1} → α → F β → F α
      H1 : LawfulFunctor F
      H2 : LawfulFunctor F
      H : ∀ (α β : Type u_1) (f : α → β) (x : F α), Eq (Functor.map f x) (Functor.ma …
      α β : Type u_1
      E1 : ∀ {α β : Type u_1}, Eq Functor.mapConst (Function.comp Functor.map (Funct …
      E2 : ∀ {α β : Type u_1}, Eq Functor.mapConst (Function.comp Functor.map (Funct …
      ⊢ Eq mc mc'
    -/
    exact E1.trans E2.symm
    /-
      🎉 no goals
    -/


/-- Introduce `id` as a quasi-functor. (Note that where a lawful `Monad` or
`Applicative` or `Functor` is needed, `Id` is the correct definition). -/
def id.mk {α : Sort u} : α → id α :=
  id


/-- `Const α` is the constant functor, mapping every type to `α`. When
`α` has a monoid structure, `Const α` has an `Applicative` instance.
(If `α` has an additive monoid structure, see `Functor.AddConst`.) -/
@[nolint unusedArguments]
def Const (α : Type*) (_β : Type*) :=
  α


/-- `Const.mk` is the canonical map `α → Const α β` (the identity), and
it can be used as a pattern to extract this value. -/
@[match_pattern]
def Const.mk {α β} (x : α) : Const α β :=
  x


/-- `Const.mk'` is `Const.mk` but specialized to map `α` to
`Const α PUnit`, where `PUnit` is the terminal object in `Type*`. -/
def Const.mk' {α} (x : α) : Const α PUnit :=
  x


/-- Extract the element of `α` from the `Const` functor. -/
def Const.run {α β} (x : Const α β) : α :=
  x


protected theorem ext {α β} {x y : Const α β} (h : x.run = y.run) : x = y :=
  h


/-- The map operation of the `Const γ` functor. -/
@[nolint unusedArguments]
protected def map {γ α β} (_f : α → β) (x : Const γ β) : Const γ α :=
  x


instance functor {γ} : Functor (Const γ) where map := @Const.map γ


                                                           /-
                                                             γ : Type u_1
                                                             ⊢ LawfulFunctor (Functor.Const γ)
                                                           -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/
instance lawfulFunctor {γ} : LawfulFunctor (Const γ) := by constructor <;> intros <;> rfl
                                                                                      /-
                                                                                        🎉 no goals
                                                                                      -/


instance {α β} [Inhabited α] : Inhabited (Const α β) :=
  ⟨(default : α)⟩


/-- `AddConst α` is a synonym for constant functor `Const α`, mapping
every type to `α`. When `α` has an additive monoid structure,
`AddConst α` has an `Applicative` instance. (If `α` has a
multiplicative monoid structure, see `Functor.Const`.) -/
def AddConst (α : Type*) :=
  Const α


/-- `AddConst.mk` is the canonical map `α → AddConst α β`, which is the identity,
where `AddConst α β = Const α β`. It can be used as a pattern to extract this value. -/
@[match_pattern]
def AddConst.mk {α β} (x : α) : AddConst α β :=
  x


/-- Extract the element of `α` from the constant functor. -/
def AddConst.run {α β} : AddConst α β → α :=
  id


instance AddConst.functor {γ} : Functor (AddConst γ) :=
  @Const.functor γ


instance AddConst.lawfulFunctor {γ} : LawfulFunctor (AddConst γ) :=
  @Const.lawfulFunctor γ


instance {α β} [Inhabited α] : Inhabited (AddConst α β) :=
  ⟨(default : α)⟩


/-- `Functor.Comp` is a wrapper around `Function.Comp` for types.
    It prevents Lean's type class resolution mechanism from trying
    a `Functor (Comp F id)` when `Functor F` would do. -/
def Comp (F : Type u → Type w) (G : Type v → Type u) (α : Type v) : Type w :=
  F <| G α


/-- Construct a term of `Comp F G α` from a term of `F (G α)`, which is the same type.
Can be used as a pattern to extract a term of `F (G α)`. -/
@[match_pattern]
def Comp.mk {F : Type u → Type w} {G : Type v → Type u} {α : Type v} (x : F (G α)) : Comp F G α :=
  x


/-- Extract a term of `F (G α)` from a term of `Comp F G α`, which is the same type. -/
def Comp.run {F : Type u → Type w} {G : Type v → Type u} {α : Type v} (x : Comp F G α) : F (G α) :=
  x


protected theorem ext {α} {x y : Comp F G α} : x.run = y.run → x = y :=
  id


instance {α} [Inhabited (F (G α))] : Inhabited (Comp F G α) :=
  ⟨(default : F (G α))⟩


/-- The map operation for the composition `Comp F G` of functors `F` and `G`. -/
protected def map {α β : Type v} (h : α → β) : Comp F G α → Comp F G β
  | Comp.mk x => Comp.mk ((h <$> ·) <$> x)


instance functor : Functor (Comp F G) where map := @Comp.map F G _ _


@[functor_norm]
theorem map_mk {α β} (h : α → β) (x : F (G α)) : h <$> Comp.mk x = Comp.mk ((h <$> ·) <$> x) :=
  rfl


@[simp]
protected theorem run_map {α β} (h : α → β) (x : Comp F G α) :
    (h <$> x).run = (h <$> ·) <$> x.run :=
  rfl


protected theorem id_map : ∀ x : Comp F G α, Comp.map id x = x
                    /-
                      F : Type u → Type w
                      G : Type v → Type u
                      inst✝³ : Functor F
                      inst✝² : Functor G
                      inst✝¹ : LawfulFunctor F
                      inst✝ : LawfulFunctor G
                      α : Type v
                      x : F (G α)
                      ⊢ Eq (Functor.Comp.map id (Functor.Comp.mk x)) (Functor.Comp.mk x)
                    -/
  | Comp.mk x => by simp only [Comp.map, id_map, id_map']; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/
  -- Porting note: `rfl` wasn't needed in mathlib3


protected theorem comp_map (g' : α → β) (h : β → γ) :
    ∀ x : Comp F G α, Comp.map (h ∘ g') x = Comp.map h (Comp.map g' x)
                    /-
                      F : Type u → Type w
                      G : Type v → Type u
                      inst✝³ : Functor F
                      inst✝² : Functor G
                      inst✝¹ : LawfulFunctor F
                      inst✝ : LawfulFunctor G
                      α β γ : Type v
                      g' : α → β
                      h : β → γ
                      x : F (G α)
                      ⊢ Eq (Functor.Comp.map (Function.comp h g') (Functor.Comp.mk x)) (Functor.Comp …
                    -/
  | Comp.mk x => by simp [Comp.map, Comp.mk, Functor.map_comp_map, functor_norm, Function.comp_def]
                    /-
                      🎉 no goals
                    -/
  -- Porting note: `Comp.mk` wasn't needed in mathlib3


instance lawfulFunctor : LawfulFunctor (Comp F G) where
  map_const := rfl
  id_map := Comp.id_map
  comp_map := Comp.comp_map

-- Porting note: had to use switch to `Id` from `id` because this has the `Functor` instance.

theorem functor_comp_id {F} [AF : Functor F] [LawfulFunctor F] :
    @Comp.functor F Id _ _ = AF :=
  @Functor.ext F _ AF (Comp.lawfulFunctor (G := Id)) _ fun _ _ _ _ => rfl

-- Porting note: had to use switch to `Id` from `id` because this has the `Functor` instance.

theorem functor_id_comp {F} [AF : Functor F] [LawfulFunctor F] : @Comp.functor Id F _ _ = AF :=
  @Functor.ext F _ AF (Comp.lawfulFunctor (F := Id)) _ fun _ _ _ _ => rfl


/-- The `<*>` operation for the composition of applicative functors. -/
protected def seq {α β : Type v} : Comp F G (α → β) → (Unit → Comp F G α) → Comp F G β
  | Comp.mk f, g => match g () with
    | Comp.mk x => Comp.mk <| (· <*> ·) <$> f <*> x
-- `ₓ` because the type of `Seq.seq` doesn't match `has_seq.seq`


instance : Pure (Comp F G) :=
  ⟨fun x => Comp.mk <| pure <| pure x⟩


instance : Seq (Comp F G) :=
  ⟨fun f x => Comp.seq f x⟩


@[simp]
protected theorem run_pure {α : Type v} : ∀ x : α, (pure x : Comp F G α).run = pure (pure x)
  | _ => rfl


@[simp]
protected theorem run_seq {α β : Type v} (f : Comp F G (α → β)) (x : Comp F G α) :
    (f <*> x).run = (· <*> ·) <$> f.run <*> x.run :=
  rfl


instance instApplicativeComp : Applicative (Comp F G) :=
  { map := @Comp.map F G _ _, seq := @Comp.seq F G _ _ }


/-- If we consider `x : F α` to, in some sense, contain values of type `α`,
predicate `Liftp p x` holds iff every value contained by `x` satisfies `p`. -/
def Liftp {α : Type u} (p : α → Prop) (x : F α) : Prop :=
  ∃ u : F (Subtype p), Subtype.val <$> u = x


/-- If we consider `x : F α` to, in some sense, contain values of type `α`, then
`Liftr r x y` relates `x` and `y` iff (1) `x` and `y` have the same shape and
(2) we can pair values `a` from `x` and `b` from `y` so that `r a b` holds. -/
def Liftr {α : Type u} (r : α → α → Prop) (x y : F α) : Prop :=
  ∃ u : F { p : α × α // r p.fst p.snd },
    (fun t : { p : α × α // r p.fst p.snd } => t.val.fst) <$> u = x ∧
      (fun t : { p : α × α // r p.fst p.snd } => t.val.snd) <$> u = y


/-- If we consider `x : F α` to, in some sense, contain values of type `α`, then
`supp x` is the set of values of type `α` that `x` contains. -/
def supp {α : Type u} (x : F α) : Set α :=
  { y : α | ∀ ⦃p⦄, Liftp p x → p y }


theorem of_mem_supp {α : Type u} {x : F α} {p : α → Prop} (h : Liftp p x) : ∀ y ∈ supp x, p y :=
  fun _ hy => hy h


/-- If `f` is a functor, if `fb : f β` and `a : α`, then `mapConstRev fb a` is the result of
  applying `f.map` to the constant function `β → α` sending everything to `a`, and then
  evaluating at `fb`. In other words it's `const a <$> fb`. -/
abbrev mapConstRev {f : Type u → Type v} [Functor f] {α β : Type u} :
    f β → α → f α :=
  fun a b => Functor.mapConst b a

/-- If `f` is a functor, if `fb : f β` and `a : α`, then `mapConstRev fb a` is the result of
  applying `f.map` to the constant function `β → α` sending everything to `a`, and then
  evaluating at `fb`. In other words it's `const a <$> fb`. -/
infix:100 " $> " => Functor.mapConstRev


