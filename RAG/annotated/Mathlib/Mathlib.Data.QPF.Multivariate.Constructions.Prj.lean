/-- The projection `i` functor -/
def Prj (v : TypeVec.{u} n) : Type u := v i


instance Prj.inhabited {v : TypeVec.{u} n} [Inhabited (v i)] : Inhabited (Prj i v) :=
  ⟨(default : v i)⟩


/-- `map` on functor `Prj i` -/
def Prj.map ⦃α β : TypeVec n⦄ (f : α ⟹ β) : Prj i α → Prj i β := f _


instance Prj.mvfunctor : MvFunctor (Prj i) where map := @Prj.map _ i


/-- Polynomial representation of the projection functor -/
def Prj.P : MvPFunctor.{u} n where
  A := PUnit
  B _ j := ULift <| PLift <| i = j


/-- Abstraction function of the `QPF` instance -/
def Prj.abs ⦃α : TypeVec n⦄ : Prj.P i α → Prj i α
  | ⟨_x, f⟩ => f _ ⟨⟨rfl⟩⟩


/-- Representation function of the `QPF` instance -/
def Prj.repr ⦃α : TypeVec n⦄ : Prj i α → Prj.P i α := fun x : α i =>
  ⟨⟨⟩, fun j ⟨⟨h⟩⟩ => (h.rec x : α j)⟩


instance Prj.mvqpf : MvQPF (Prj i) where
  P := Prj.P i
  abs := @Prj.abs _ i
  repr := @Prj.repr _ i
                 /-
                   n : Nat
                   i : Fin2 n
                   ⊢ ∀ {α : TypeVec.{?u.879} n} (x : MvQPF.Prj i α), Eq (MvQPF.Prj.abs i (MvQPF.P …
                 -/
  abs_repr := by intros; rfl
                         /-
                           🎉 no goals
                         -/
                /-
                  n : Nat
                  i : Fin2 n
                  ⊢ ∀ {α β : TypeVec.{?u.879} n} (f : α.Arrow β) (p : ↑(MvQPF.Prj.P i) α), Eq (M …
                -/
  abs_map := by intros α β f P; cases P; rfl
                                         /-
                                           🎉 no goals
                                         -/


