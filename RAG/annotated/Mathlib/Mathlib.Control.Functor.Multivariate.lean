/-- Multivariate functors, i.e. functor between the category of type vectors
and the category of Type -/
class MvFunctor {n : ℕ} (F : TypeVec n → Type*) where
  /-- Multivariate map, if `f : α ⟹ β` and `x : F α` then `f <$$> x : F β`. -/
  map : ∀ {α β : TypeVec n}, α ⟹ β → F α → F β


/-- Multivariate map, if `f : α ⟹ β` and `x : F α` then `f <$$> x : F β` -/
scoped[MvFunctor] infixr:100 " <$$> " => MvFunctor.map


/-- predicate lifting over multivariate functors -/
def LiftP {α : TypeVec n} (P : ∀ i, α i → Prop) (x : F α) : Prop :=
  ∃ u : F (fun i => Subtype (P i)), (fun i => @Subtype.val _ (P i)) <$$> u = x


/-- relational lifting over multivariate functors -/
def LiftR {α : TypeVec n} (R : ∀ {i}, α i → α i → Prop) (x y : F α) : Prop :=
  ∃ u : F (fun i => { p : α i × α i // R p.fst p.snd }),
    (fun i (t : { p : α i × α i // R p.fst p.snd }) => t.val.fst) <$$> u = x ∧
      (fun i (t : { p : α i × α i // R p.fst p.snd }) => t.val.snd) <$$> u = y


/-- given `x : F α` and a projection `i` of type vector `α`, `supp x i` is the set
of `α.i` contained in `x` -/
def supp {α : TypeVec n} (x : F α) (i : Fin2 n) : Set (α i) :=
  { y : α i | ∀ ⦃P⦄, LiftP P x → P i y }


theorem of_mem_supp {α : TypeVec n} {x : F α} {P : ∀ ⦃i⦄, α i → Prop} (h : LiftP P x) (i : Fin2 n) :
    ∀ y ∈ supp x i, P y := fun _y hy => hy h


/-- laws for `MvFunctor` -/
class LawfulMvFunctor {n : ℕ} (F : TypeVec n → Type*) [MvFunctor F] : Prop where
  /-- `map` preserved identities, i.e., maps identity on `α` to identity on `F α` -/
  id_map : ∀ {α : TypeVec n} (x : F α), TypeVec.id <$$> x = x
  /-- `map` preserves compositions -/
  comp_map :
    ∀ {α β γ : TypeVec n} (g : α ⟹ β) (h : β ⟹ γ) (x : F α), (h ⊚ g) <$$> x = h <$$> g <$$> x


/-- adapt `MvFunctor.LiftP` to accept predicates as arrows -/
def LiftP' : F α → Prop :=
  MvFunctor.LiftP fun i x => ofRepeat <| P i x



/-- adapt `MvFunctor.LiftR` to accept relations as arrows -/
def LiftR' : F α → F α → Prop :=
  MvFunctor.LiftR @fun i x y => ofRepeat <| R i <| TypeVec.prod.mk _ x y


@[simp]
theorem id_map (x : F α) : TypeVec.id <$$> x = x :=
  LawfulMvFunctor.id_map x


@[simp]
theorem id_map' (x : F α) : (fun _i a => a) <$$> x = x :=
  id_map x


theorem map_map (g : α ⟹ β) (h : β ⟹ γ) (x : F α) : h <$$> g <$$> x = (h ⊚ g) <$$> x :=
  Eq.symm <| comp_map _ _ _


theorem exists_iff_exists_of_mono {P : F α → Prop} {q : F β → Prop}
    (f : α ⟹ β) (g : β ⟹ α)
    (h₀ : f ⊚ g = TypeVec.id)
    (h₁ : ∀ u : F α, P u ↔ q (f <$$> u)) :
    (∃ u : F α, P u) ↔ ∃ u : F β, q u := by
  /-
    n : Nat
    α β : TypeVec.{u} n
    F : TypeVec.{u} n → Type v
    inst✝¹ : MvFunctor F
    inst✝ : LawfulMvFunctor F
    P : F α → Prop
    q : F β → Prop
    f : α.Arrow β
    g : β.Arrow α
    h₀ : Eq (TypeVec.comp f g) TypeVec.id
    h₁ : ∀ (u : F α), Iff (P u) (q (MvFunctor.map f u))
    ⊢ Iff (Exists fun u => P u) (Exists fun u => q u)
  -/
  constructor <;> rintro ⟨u, h₂⟩
    /-
      case mp.intro
      n : Nat
      α β : TypeVec.{u} n
      F : TypeVec.{u} n → Type v
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      P : F α → Prop
      q : F β → Prop
      f : α.Arrow β
      g : β.Arrow α
      h₀ : Eq (TypeVec.comp f g) TypeVec.id
      h₁ : ∀ (u : F α), Iff (P u) (q (MvFunctor.map f u))
      u : F α
      h₂ : P u
      ⊢ Exists fun u => q u
    -/
  · refine ⟨f <$$> u, ?_⟩
    /-
      case mp.intro
      n : Nat
      α β : TypeVec.{u} n
      F : TypeVec.{u} n → Type v
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      P : F α → Prop
      q : F β → Prop
      f : α.Arrow β
      g : β.Arrow α
      h₀ : Eq (TypeVec.comp f g) TypeVec.id
      h₁ : ∀ (u : F α), Iff (P u) (q (MvFunctor.map f u))
      u : F α
      h₂ : P u
      ⊢ q (MvFunctor.map f u)
    -/
    apply (h₁ u).mp h₂
    /-
      🎉 no goals
    -/
    /-
      case mpr.intro
      n : Nat
      α β : TypeVec.{u} n
      F : TypeVec.{u} n → Type v
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      P : F α → Prop
      q : F β → Prop
      f : α.Arrow β
      g : β.Arrow α
      h₀ : Eq (TypeVec.comp f g) TypeVec.id
      h₁ : ∀ (u : F α), Iff (P u) (q (MvFunctor.map f u))
      u : F β
      h₂ : q u
      ⊢ Exists fun u => P u
    -/
  · refine ⟨g <$$> u, ?_⟩
    /-
      case mpr.intro
      n : Nat
      α β : TypeVec.{u} n
      F : TypeVec.{u} n → Type v
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      P : F α → Prop
      q : F β → Prop
      f : α.Arrow β
      g : β.Arrow α
      h₀ : Eq (TypeVec.comp f g) TypeVec.id
      h₁ : ∀ (u : F α), Iff (P u) (q (MvFunctor.map f u))
      u : F β
      h₂ : q u
      ⊢ P (MvFunctor.map g u)
    -/
    rw [h₁]
    /-
      case mpr.intro
      n : Nat
      α β : TypeVec.{u} n
      F : TypeVec.{u} n → Type v
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      P : F α → Prop
      q : F β → Prop
      f : α.Arrow β
      g : β.Arrow α
      h₀ : Eq (TypeVec.comp f g) TypeVec.id
      h₁ : ∀ (u : F α), Iff (P u) (q (MvFunctor.map f u))
      u : F β
      h₂ : q u
      ⊢ q (MvFunctor.map f (MvFunctor.map g u))
    -/
    simp only [MvFunctor.map_map, h₀, LawfulMvFunctor.id_map, h₂]
    /-
      🎉 no goals
    -/


theorem LiftP_def (x : F α) : LiftP' P x ↔ ∃ u : F (Subtype_ P), subtypeVal P <$$> u = x :=
                                                               /-
                                                                 n : Nat
                                                                 α : TypeVec.{u} n
                                                                 F : TypeVec.{u} n → Type v
                                                                 inst✝¹ : MvFunctor F
                                                                 P : α.Arrow (TypeVec.repeat n Prop)
                                                                 inst✝ : LawfulMvFunctor F
                                                                 x : F α
                                                                 ⊢ ∀ (u : F fun i => Subtype ((fun i x => TypeVec.ofRepeat (P i x)) i)), Iff (E …
                                                               -/
  exists_iff_exists_of_mono F _ _ (toSubtype_of_subtype P) (by simp [MvFunctor.map_map])
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem LiftR_def (x y : F α) :
    LiftR' R x y ↔
      ∃ u : F (Subtype_ R),
        (TypeVec.prod.fst ⊚ subtypeVal R) <$$> u = x ∧
          (TypeVec.prod.snd ⊚ subtypeVal R) <$$> u = y :=
  exists_iff_exists_of_mono _ _ _ (toSubtype'_of_subtype' R) (by
    /-
      n : Nat
      α : TypeVec.{u} n
      F : TypeVec.{u} n → Type v
      inst✝¹ : MvFunctor F
      R : (α.prod α).Arrow (TypeVec.repeat n Prop)
      inst✝ : LawfulMvFunctor F
      x y : F α
      ⊢ ∀ (u : F fun i => Subtype fun p => (fun i x y => TypeVec.ofRepeat (R i (Type …
    -/
    simp only [map_map, comp_assoc, subtypeVal_toSubtype']
    /-
      n : Nat
      α : TypeVec.{u} n
      F : TypeVec.{u} n → Type v
      inst✝¹ : MvFunctor F
      R : (α.prod α).Arrow (TypeVec.repeat n Prop)
      inst✝ : LawfulMvFunctor F
      x y : F α
      ⊢ ∀ (u : F fun i => Subtype fun p => TypeVec.ofRepeat (R i (TypeVec.prod.mk i  …
    -/
    simp (config := { unfoldPartialApp := true }) [comp])
    /-
      🎉 no goals
    -/


private def f :
    ∀ n α,
      (fun i : Fin2 (n + 1) => { p_1 // ofRepeat (PredLast' α pp i p_1) }) ⟹ fun i : Fin2 (n + 1) =>
        { p_1 : (α ::: β) i // PredLast α pp p_1 }
  | _, α, Fin2.fs i, x =>
                     /-
                       n : Nat
                       F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
                       inst✝¹ : MvFunctor F
                       inst✝ : LawfulMvFunctor F
                       α✝ : TypeVec.{u} n
                       β : Type u
                       pp : β → Prop
                       x✝ : Nat
                       α : TypeVec.{u} x✝
                       i : Fin2 x✝
                       x : (fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.PredLast' pp i p_1)) i.fs
                       ⊢ Eq (TypeVec.ofRepeat (α.PredLast' pp i.fs ↑x)) (α.PredLast pp ↑x)
                     -/
    ⟨x.val, cast (by simp only [PredLast]; erw [const_iff_true]) x.property⟩
                                           /-
                                             🎉 no goals
                                           -/
  | _, _, Fin2.fz, x => ⟨x.val, x.property⟩


private def g :
    ∀ n α,
      (fun i : Fin2 (n + 1) => { p_1 : (α ::: β) i // PredLast α pp p_1 }) ⟹ fun i : Fin2 (n + 1) =>
        { p_1 // ofRepeat (PredLast' α pp i p_1) }
  | _, α, Fin2.fs i, x =>
                     /-
                       n : Nat
                       F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
                       inst✝¹ : MvFunctor F
                       inst✝ : LawfulMvFunctor F
                       α✝ : TypeVec.{u} n
                       β : Type u
                       pp : β → Prop
                       x✝ : Nat
                       α : TypeVec.{u} x✝
                       i : Fin2 x✝
                       x : (fun i => Subtype fun p_1 => α.PredLast pp p_1) i.fs
                       ⊢ Eq (α.PredLast pp ↑x) (TypeVec.ofRepeat (α.PredLast' pp i.fs ↑x))
                     -/
    ⟨x.val, cast (by simp only [PredLast]; erw [const_iff_true]) x.property⟩
                                           /-
                                             🎉 no goals
                                           -/
  | _, _, Fin2.fz, x => ⟨x.val, x.property⟩


theorem LiftP_PredLast_iff {β} (P : β → Prop) (x : F (α ::: β)) :
    LiftP' (PredLast' _ P) x ↔ LiftP (PredLast _ P) x := by
  /-
    n : Nat
    F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
    inst✝¹ : MvFunctor F
    inst✝ : LawfulMvFunctor F
    α : TypeVec.{u} n
    β : Type u
    P : β → Prop
    x : F (α.append1 β)
    ⊢ Iff (MvFunctor.LiftP' (α.PredLast' P) x) (MvFunctor.LiftP (α.PredLast P) x)
  -/
  dsimp only [LiftP, LiftP']
  /-
    n : Nat
    F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
    inst✝¹ : MvFunctor F
    inst✝ : LawfulMvFunctor F
    α : TypeVec.{u} n
    β : Type u
    P : β → Prop
    x : F (α.append1 β)
    ⊢ Iff (Exists fun u => Eq (MvFunctor.map (fun i => Subtype.val) u) x) (Exists  …
  -/
  apply exists_iff_exists_of_mono F (f _ n α) (g _ n α)
    /-
      case h₀
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x : F (α.append1 β)
      ⊢ Eq (TypeVec.comp (MvFunctor.f P n α) (MvFunctor.g P n α)) TypeVec.id
    -/
  · ext i ⟨x, _⟩
    /-
      case h₀.a.h.mk.a
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x✝ : F (α.append1 β)
      i : Fin2 (HAdd.hAdd n 1)
      x : α.append1 β i
      property✝ : α.PredLast P x
      ⊢ Eq ↑(TypeVec.comp (MvFunctor.f P n α) (MvFunctor.g P n α) i ⟨x, property✝⟩)  …
    -/
                /-
                  🎉 no goals
                -/
    cases i <;> rfl
                /-
                  🎉 no goals
                -/
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x : F (α.append1 β)
      ⊢ ∀ (u : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.PredLast' P i p_1)) …
    -/
  · intros
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x : F (α.append1 β)
      u✝ : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.PredLast' P i p_1)
      ⊢ Iff (Eq (MvFunctor.map (fun i => Subtype.val) u✝) x) (Eq (MvFunctor.map (fun …
    -/
    rw [MvFunctor.map_map]
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x : F (α.append1 β)
      u✝ : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.PredLast' P i p_1)
      ⊢ Iff (Eq (MvFunctor.map (fun i => Subtype.val) u✝) x) (Eq (MvFunctor.map (Typ …
    -/
    dsimp (config := { unfoldPartialApp := true }) [(· ⊚ ·)]
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x : F (α.append1 β)
      u✝ : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.PredLast' P i p_1)
      ⊢ Iff (Eq (MvFunctor.map (fun i => Subtype.val) u✝) x) (Eq (MvFunctor.map (fun …
    -/
    suffices (fun i => Subtype.val) = (fun i x => (MvFunctor.f P n α i x).val) by rw [this]
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x : F (α.append1 β)
      u✝ : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.PredLast' P i p_1)
      ⊢ Eq (fun i => Subtype.val) fun i x => ↑(MvFunctor.f P n α i x)
    -/
    ext i ⟨x, _⟩
    /-
      case h₁.h.h.mk
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      P : β → Prop
      x✝ : F (α.append1 β)
      u✝ : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.PredLast' P i p_1)
      i : Fin2 (HAdd.hAdd n 1)
      x : α.append1 β i
      property✝ : TypeVec.ofRepeat (α.PredLast' P i x)
      ⊢ Eq ↑⟨x, property✝⟩ ↑(MvFunctor.f P n α i ⟨x, property✝⟩)
    -/
                /-
                  🎉 no goals
                -/
    cases i <;> rfl
                /-
                  🎉 no goals
                -/


private def f' :
    ∀ n α,
      (fun i : Fin2 (n + 1) =>
          { p_1 : _ × _ // ofRepeat (RelLast' α rr i (TypeVec.prod.mk _ p_1.fst p_1.snd)) }) ⟹
        fun i : Fin2 (n + 1) => { p_1 : (α ::: β) i × _ // RelLast α rr p_1.fst p_1.snd }
  | _, α, Fin2.fs i, x =>
                     /-
                       n : Nat
                       F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
                       inst✝¹ : MvFunctor F
                       inst✝ : LawfulMvFunctor F
                       α✝ : TypeVec.{u} n
                       β : Type u
                       pp : β → Prop
                       rr : β → β → Prop
                       x✝ : Nat
                       α : TypeVec.{u} x✝
                       i : Fin2 x✝
                       x : (fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.RelLast' rr i (TypeVec.pr …
                       ⊢ Eq (TypeVec.ofRepeat (α.RelLast' rr i.fs (TypeVec.prod.mk i.fs (↑x).1 (↑x).2 …
                     -/
    ⟨x.val, cast (by simp only [RelLast]; erw [repeatEq_iff_eq]) x.property⟩
                                          /-
                                            🎉 no goals
                                          -/
  | _, _, Fin2.fz, x => ⟨x.val, x.property⟩


private def g' :
    ∀ n α,
      (fun i : Fin2 (n + 1) => { p_1 : (α ::: β) i × _ // RelLast α rr p_1.fst p_1.snd }) ⟹
        fun i : Fin2 (n + 1) =>
        { p_1 : _ × _ // ofRepeat (RelLast' α rr i (TypeVec.prod.mk _ p_1.1 p_1.2)) }
  | _, α, Fin2.fs i, x =>
                     /-
                       n : Nat
                       F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
                       inst✝¹ : MvFunctor F
                       inst✝ : LawfulMvFunctor F
                       α✝ : TypeVec.{u} n
                       β : Type u
                       pp : β → Prop
                       rr : β → β → Prop
                       x✝ : Nat
                       α : TypeVec.{u} x✝
                       i : Fin2 x✝
                       x : (fun i => Subtype fun p_1 => α.RelLast rr p_1.1 p_1.2) i.fs
                       ⊢ Eq (α.RelLast rr (↑x).1 (↑x).2) (TypeVec.ofRepeat (α.RelLast' rr i.fs (TypeV …
                     -/
    ⟨x.val, cast (by simp only [RelLast]; erw [repeatEq_iff_eq]) x.property⟩
                                          /-
                                            🎉 no goals
                                          -/
  | _, _, Fin2.fz, x => ⟨x.val, x.property⟩


theorem LiftR_RelLast_iff (x y : F (α ::: β)) :
    LiftR' (RelLast' _ rr) x y ↔ LiftR (RelLast (i := _) _ rr) x y := by
  /-
    n : Nat
    F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
    inst✝¹ : MvFunctor F
    inst✝ : LawfulMvFunctor F
    α : TypeVec.{u} n
    β : Type u
    rr : β → β → Prop
    x y : F (α.append1 β)
    ⊢ Iff (MvFunctor.LiftR' (α.RelLast' rr) x y) (MvFunctor.LiftR (fun {i} => α.Re …
  -/
  dsimp only [LiftR, LiftR']
  /-
    n : Nat
    F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
    inst✝¹ : MvFunctor F
    inst✝ : LawfulMvFunctor F
    α : TypeVec.{u} n
    β : Type u
    rr : β → β → Prop
    x y : F (α.append1 β)
    ⊢ Iff (Exists fun u => And (Eq (MvFunctor.map (fun i t => (↑t).1) u) x) (Eq (M …
  -/
  apply exists_iff_exists_of_mono F (f' rr _ _) (g' rr _ _)
    /-
      case h₀
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      rr : β → β → Prop
      x y : F (α.append1 β)
      ⊢ Eq (TypeVec.comp (MvFunctor.f' rr n α) (MvFunctor.g' rr n α)) TypeVec.id
    -/
  · ext i ⟨x, _⟩ : 2
    /-
      case h₀.a.h.mk
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      rr : β → β → Prop
      x✝ y : F (α.append1 β)
      i : Fin2 (HAdd.hAdd n 1)
      x : Prod (α.append1 β i) (α.append1 β i)
      property✝ : α.RelLast rr x.1 x.2
      ⊢ Eq (TypeVec.comp (MvFunctor.f' rr n α) (MvFunctor.g' rr n α) i ⟨x, property✝ …
    -/
                /-
                  🎉 no goals
                -/
    cases i <;> rfl
                /-
                  🎉 no goals
                -/
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      rr : β → β → Prop
      x y : F (α.append1 β)
      ⊢ ∀ (u : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.RelLast' rr i (Type …
    -/
  · intros
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      rr : β → β → Prop
      x y : F (α.append1 β)
      u✝ : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.RelLast' rr i (TypeVec. …
      ⊢ Iff (And (Eq (MvFunctor.map (fun i t => (↑t).1) u✝) x) (Eq (MvFunctor.map (f …
    -/
    simp (config := { unfoldPartialApp := true }) only [map_map, TypeVec.comp]
    -- Porting note: proof was
    -- rw [MvFunctor.map_map, MvFunctor.map_map, (· ⊚ ·), (· ⊚ ·)]
    -- congr <;> ext i ⟨x, _⟩ <;> cases i <;> rfl
    suffices (fun i t => t.val.fst) = ((fun i x => (MvFunctor.f' rr n α i x).val.fst))
            ∧ (fun i t => t.val.snd) = ((fun i x => (MvFunctor.f' rr n α i x).val.snd)) by
      rw [this.1, this.2]
    /-
      case h₁
      n : Nat
      F : TypeVec.{u} (HAdd.hAdd n 1) → Type u_1
      inst✝¹ : MvFunctor F
      inst✝ : LawfulMvFunctor F
      α : TypeVec.{u} n
      β : Type u
      rr : β → β → Prop
      x y : F (α.append1 β)
      u✝ : F fun i => Subtype fun p_1 => TypeVec.ofRepeat (α.RelLast' rr i (TypeVec. …
      ⊢ And (Eq (fun i t => (↑t).1) fun i x => (↑(MvFunctor.f' rr n α i x)).1) (Eq ( …
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
    constructor <;> ext i ⟨x, _⟩ <;> cases i <;> rfl
                                                 /-
                                                   🎉 no goals
                                                 -/


/-- Any type function that is (extensionally) equivalent to a functor, is itself a functor -/
def ofEquiv {F F' : TypeVec.{u} n → Type*} [MvFunctor F'] (eqv : ∀ α, F α ≃ F' α) :
    MvFunctor F where
  map f x := (eqv _).symm <| f <$$> eqv _ x


