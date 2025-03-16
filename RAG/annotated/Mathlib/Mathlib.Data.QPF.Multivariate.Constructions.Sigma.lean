/-- Dependent sum of an `n`-ary functor. The sum can range over
data types like `ℕ` or over `Type.{u-1}` -/
def Sigma (v : TypeVec.{u} n) : Type u :=
  Σ α : A, F α v


/-- Dependent product of an `n`-ary functor. The sum can range over
data types like `ℕ` or over `Type.{u-1}` -/
def Pi (v : TypeVec.{u} n) : Type u :=
  ∀ α : A, F α v


instance Sigma.inhabited {α} [Inhabited A] [Inhabited (F default α)] : Inhabited (Sigma F α) :=
  ⟨⟨default, default⟩⟩


instance Pi.inhabited {α} [∀ a, Inhabited (F a α)] : Inhabited (Pi F α) :=
  ⟨fun _a => default⟩


instance [∀ α, MvFunctor <| F α] : MvFunctor (Sigma F) where
  map := fun f ⟨a, x⟩ => ⟨a, f <$$> x⟩



/-- polynomial functor representation of a dependent sum -/
protected def P : MvPFunctor n :=
  ⟨Σ a, (P (F a)).A, fun x => (P (F x.1)).B x.2⟩


/-- abstraction function for dependent sums -/
protected def abs ⦃α⦄ : Sigma.P F α → Sigma F α
  | ⟨a, f⟩ => ⟨a.1, MvQPF.abs ⟨a.2, f⟩⟩


/-- representation function for dependent sums -/
protected def repr ⦃α⦄ : Sigma F α → Sigma.P F α
  | ⟨a, f⟩ =>
    let x := MvQPF.repr f
    ⟨⟨a, x.1⟩, x.2⟩


instance : MvQPF (Sigma F) where
  P := Sigma.P F
  abs {α} := @Sigma.abs _ _ F _ α
  repr {α} := @Sigma.repr _ _ F _ α
                 /-
                   n : Nat
                   A : Type u
                   F : A → TypeVec.{u} n → Type u
                   inst✝ : (α : A) → MvQPF (F α)
                   ⊢ ∀ {α : TypeVec.{u} n} (x : MvQPF.Sigma F α), Eq ((fun {α} => MvQPF.Sigma.abs …
                 -/
  abs_repr := by rintro α ⟨x, f⟩; simp only [Sigma.abs, Sigma.repr, Sigma.eta, abs_repr]
                                  /-
                                    🎉 no goals
                                  -/
                /-
                  n : Nat
                  A : Type u
                  F : A → TypeVec.{u} n → Type u
                  inst✝ : (α : A) → MvQPF (F α)
                  ⊢ ∀ {α β : TypeVec.{u} n} (f : α.Arrow β) (p : ↑(MvQPF.Sigma.P F) α), Eq ((fun …
                -/
  abs_map := by rintro α β f ⟨x, g⟩; simp only [Sigma.abs, MvPFunctor.map_eq]
                /-
                  case mk
                  n : Nat
                  A : Type u
                  F : A → TypeVec.{u} n → Type u
                  inst✝ : (α : A) → MvQPF (F α)
                  α β : TypeVec.{u} n
                  f : α.Arrow β
                  x : (MvQPF.Sigma.P F).A
                  g : ((MvQPF.Sigma.P F).B x).Arrow α
                  ⊢ Eq ⟨x.fst, MvQPF.abs ⟨x.snd, TypeVec.comp f g⟩⟩ (MvFunctor.map f ⟨x.fst, MvQ …
                -/
                simp only [(· <$$> ·), ← abs_map, ← MvPFunctor.map_eq]
                /-
                  🎉 no goals
                -/


instance [∀ α, MvFunctor <| F α] : MvFunctor (Pi F) where map f x a := f <$$> x a


/-- polynomial functor representation of a dependent product -/
protected def P : MvPFunctor n :=
  ⟨∀ a, (P (F a)).A, fun x i => Σ a, (P (F a)).B (x a) i⟩


/-- abstraction function for dependent products -/
protected def abs ⦃α⦄ : Pi.P F α → Pi F α
  | ⟨a, f⟩ => fun x => MvQPF.abs ⟨a x, fun i y => f i ⟨_, y⟩⟩


/-- representation function for dependent products -/
protected def repr ⦃α⦄ : Pi F α → Pi.P F α
  | f => ⟨fun a => (MvQPF.repr (f a)).1, fun _i a => (MvQPF.repr (f _)).2 _ a.2⟩


instance : MvQPF (Pi F) where
  P := Pi.P F
  abs := @Pi.abs _ _ F _
  repr := @Pi.repr _ _ F _
                 /-
                   n : Nat
                   A : Type u
                   F : A → TypeVec.{u} n → Type u
                   inst✝ : (α : A) → MvQPF (F α)
                   ⊢ ∀ {α : TypeVec.{u} n} (x : MvQPF.Pi F α), Eq (MvQPF.Pi.abs F (MvQPF.Pi.repr  …
                 -/
  abs_repr := by rintro α f; simp only [Pi.abs, Pi.repr, Sigma.eta, abs_repr]
                             /-
                               🎉 no goals
                             -/
                /-
                  n : Nat
                  A : Type u
                  F : A → TypeVec.{u} n → Type u
                  inst✝ : (α : A) → MvQPF (F α)
                  ⊢ ∀ {α β : TypeVec.{u} n} (f : α.Arrow β) (p : ↑(MvQPF.Pi.P F) α), Eq (MvQPF.P …
                -/
  abs_map := by rintro α β f ⟨x, g⟩; simp only [Pi.abs, (· <$$> ·), ← abs_map]; rfl
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


