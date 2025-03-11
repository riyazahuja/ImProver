/-- If `F` is a QPF then `G` is a QPF as well. Can be used to
construct `MvQPF` instances by transporting them across
surjective functions -/
def quotientQPF (FG_abs_repr : ∀ {α} (x : G α), FG_abs (FG_repr x) = x)
    (FG_abs_map : ∀ {α β} (f : α ⟹ β) (x : F α), FG_abs (f <$$> x) = f <$$> FG_abs x) :
    MvQPF G where
  P := q.P
  abs p := FG_abs (abs p)
  repr x := repr (FG_repr x)
                   /-
                     n : Nat
                     F : TypeVec.{u} n → Type u
                     q : MvQPF F
                     G : TypeVec.{u} n → Type u
                     inst✝ : MvFunctor G
                     FG_abs : {α : TypeVec.{u} n} → F α → G α
                     FG_repr : {α : TypeVec.{u} n} → G α → F α
                     FG_abs_repr : ∀ {α : TypeVec.{u} n} (x : G α), Eq (FG_abs (FG_repr x)) x
                     FG_abs_map : ∀ {α β : TypeVec.{u} n} (f : α.Arrow β) (x : F α), Eq (FG_abs (Mv …
                     α✝ : TypeVec.{u} n
                     x : G α✝
                     ⊢ Eq ((fun {α} p => FG_abs (MvQPF.abs p)) ((fun {α} x => MvQPF.repr (FG_repr x …
                   -/
  abs_repr x := by dsimp; rw [abs_repr, FG_abs_repr]
                          /-
                            🎉 no goals
                          -/
                    /-
                      n : Nat
                      F : TypeVec.{u} n → Type u
                      q : MvQPF F
                      G : TypeVec.{u} n → Type u
                      inst✝ : MvFunctor G
                      FG_abs : {α : TypeVec.{u} n} → F α → G α
                      FG_repr : {α : TypeVec.{u} n} → G α → F α
                      FG_abs_repr : ∀ {α : TypeVec.{u} n} (x : G α), Eq (FG_abs (FG_repr x)) x
                      FG_abs_map : ∀ {α β : TypeVec.{u} n} (f : α.Arrow β) (x : F α), Eq (FG_abs (Mv …
                      α✝ β✝ : TypeVec.{u} n
                      f : α✝.Arrow β✝
                      p : ↑(MvQPF.P F) α✝
                      ⊢ Eq ((fun {α} p => FG_abs (MvQPF.abs p)) (MvFunctor.map f p)) (MvFunctor.map  …
                    -/
  abs_map f p := by dsimp; rw [abs_map, FG_abs_map]
                           /-
                             🎉 no goals
                           -/


/-- Functorial quotient type -/
def Quot1 (α : TypeVec n) :=
  Quot (@R α)


instance Quot1.inhabited {α : TypeVec n} [Inhabited <| F α] : Inhabited (Quot1 R α) :=
  ⟨Quot.mk _ default⟩


/-- `map` of the `Quot1` functor -/
def Quot1.map ⦃α β⦄ (f : α ⟹ β) : Quot1.{u} R α → Quot1.{u} R β :=
  Quot.lift (fun x : F α => Quot.mk _ (f <$$> x : F β)) fun a b h => Quot.sound <| Hfunc a b _ h


/-- `mvFunctor` instance for `Quot1` with well-behaved `R` -/
def Quot1.mvFunctor : MvFunctor (Quot1 R) where map := @Quot1.map _ _ R _ Hfunc


/-- `Quot1` is a QPF -/
noncomputable def relQuot : @MvQPF _ (Quot1 R) :=
  @quotientQPF n F q _ (MvQPF.Quot1.mvFunctor R Hfunc) (fun x => Quot.mk _ x)
    Quot.out (fun _x => Quot.out_eq _) fun _f _x => rfl


