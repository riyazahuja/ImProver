/-- Composition of an `n`-ary functor with `n` `m`-ary
functors gives us one `m`-ary functor -/
def Comp (v : TypeVec.{u} m) : Type _ :=
  F fun i : Fin2 n ↦ G i v


instance [I : Inhabited (F fun i : Fin2 n ↦ G i α)] : Inhabited (Comp F G α) := I


/-- Constructor for functor composition -/
protected def mk (x : F fun i ↦ G i α) : Comp F G α := x


/-- Destructor for functor composition -/
protected def get (x : Comp F G α) : F fun i ↦ G i α := x


@[simp]
protected theorem mk_get (x : Comp F G α) : Comp.mk (Comp.get x) = x := rfl


@[simp]
protected theorem get_mk (x : F fun i ↦ G i α) : Comp.get (Comp.mk x) = x := rfl


/-- map operation defined on a vector of functors -/
protected def map' : (fun i : Fin2 n ↦ G i α) ⟹ fun i : Fin2 n ↦ G i β := fun _i ↦ map f


/-- The composition of functors is itself functorial -/
protected def map : (Comp F G) α → (Comp F G) β :=
  (map fun _i ↦ map f : (F fun i ↦ G i α) → F fun i ↦ G i β)


instance : MvFunctor (Comp F G) where map f := Comp.map f


theorem map_mk (x : F fun i ↦ G i α) :
    f <$$> Comp.mk x = Comp.mk ((fun i (x : G i α) ↦ f <$$> x) <$$> x) := rfl


theorem get_map (x : Comp F G α) :
    Comp.get (f <$$> x) = (fun i (x : G i α) ↦ f <$$> x) <$$> Comp.get x := rfl


instance [MvQPF F] [∀ i, MvQPF <| G i] : MvQPF (Comp F G) where
  P := MvPFunctor.comp (P F) fun i ↦ P <| G i
  abs := Comp.mk ∘ (map fun _ ↦ abs) ∘ abs ∘ MvPFunctor.comp.get
  repr {α} := MvPFunctor.comp.mk ∘ repr ∘
              (map fun i ↦ (repr : G i α → (fun i : Fin2 n ↦ Obj (P (G i)) α) i)) ∘ Comp.get
  abs_repr := by
    /-
      n m : Nat
      F : TypeVec.{u} n → Type u_1
      G : Fin2 n → TypeVec.{u} m → Type u
      α β : TypeVec.{u} m
      f : α.Arrow β
      inst✝¹ : MvQPF F
      inst✝ : (i : Fin2 n) → MvQPF (G i)
      ⊢ ∀ {α : TypeVec.{u} m} (x : MvQPF.Comp F G α), Eq ((fun {α} => Function.comp  …
    -/
    intros
    simp (config := { unfoldPartialApp := true }) only [Function.comp_def, comp.get_mk, abs_repr,
      map_map, TypeVec.comp, MvFunctor.id_map', Comp.mk_get]
  abs_map := by
    /-
      n m : Nat
      F : TypeVec.{u} n → Type u_1
      G : Fin2 n → TypeVec.{u} m → Type u
      α β : TypeVec.{u} m
      f : α.Arrow β
      inst✝¹ : MvQPF F
      inst✝ : (i : Fin2 n) → MvQPF (G i)
      ⊢ ∀ {α β : TypeVec.{u} m} (f : α.Arrow β) (p : ↑((MvQPF.P F).comp fun i => MvQ …
    -/
    intros
    /-
      n m : Nat
      F : TypeVec.{u} n → Type u_1
      G : Fin2 n → TypeVec.{u} m → Type u
      α β : TypeVec.{u} m
      f : α.Arrow β
      inst✝¹ : MvQPF F
      inst✝ : (i : Fin2 n) → MvQPF (G i)
      α✝ β✝ : TypeVec.{u} m
      f✝ : α✝.Arrow β✝
      p✝ : ↑((MvQPF.P F).comp fun i => MvQPF.P (G i)) α✝
      ⊢ Eq ((fun {α} => Function.comp MvQPF.Comp.mk (Function.comp (MvFunctor.map fu …
    -/
    simp only [(· ∘ ·)]
    /-
      n m : Nat
      F : TypeVec.{u} n → Type u_1
      G : Fin2 n → TypeVec.{u} m → Type u
      α β : TypeVec.{u} m
      f : α.Arrow β
      inst✝¹ : MvQPF F
      inst✝ : (i : Fin2 n) → MvQPF (G i)
      α✝ β✝ : TypeVec.{u} m
      f✝ : α✝.Arrow β✝
      p✝ : ↑((MvQPF.P F).comp fun i => MvQPF.P (G i)) α✝
      ⊢ Eq (MvQPF.Comp.mk (MvFunctor.map (fun x => MvQPF.abs) (MvQPF.abs (MvPFunctor …
    -/
    rw [← abs_map]
    simp (config := { unfoldPartialApp := true }) only [comp.get_map, map_map, TypeVec.comp,
      abs_map, map_mk]


