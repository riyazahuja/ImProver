/-- A path from the root of a tree to one of its node -/
inductive M.Path : P.last.M → Fin2 n → Type u
  | root (x : P.last.M)
          (a : P.A)
          (f : P.last.B a → P.last.M)
          (h : PFunctor.M.dest x = ⟨a, f⟩)
          (i : Fin2 n)
          (c : P.drop.B a i) : M.Path x i
  | child (x : P.last.M)
          (a : P.A)
          (f : P.last.B a → P.last.M)
          (h : PFunctor.M.dest x = ⟨a, f⟩)
          (j : P.last.B a)
          (i : Fin2 n)
          (c : M.Path (f j) i) : M.Path x i


instance M.Path.inhabited (x : P.last.M) {i} [Inhabited (P.drop.B x.head i)] :
    Inhabited (M.Path P x i) :=
  let a := PFunctor.M.head x
  let f := PFunctor.M.children x
  ⟨M.Path.root _ a f
      (PFunctor.M.casesOn' x
        (r := fun _ => PFunctor.M.dest x = ⟨a, f⟩)
        <| by
        /-
          n : Nat
          P : MvPFunctor.{u} (HAdd.hAdd n 1)
          x : P.last.M
          i : Fin2 n
          inst✝ : Inhabited (P.drop.B x.head i)
          a : P.last.A := x.head
          f : P.last.B x.head → P.last.M := x.children
          ⊢ ∀ (a_1 : P.last.A) (f_1 : P.last.B a_1 → P.last.M), (fun x_1 => Eq x.dest ⟨a …
        -/
        intros; simp [a, PFunctor.M.dest_mk, PFunctor.M.children_mk]; rfl)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
      _ default⟩


/-- Polynomial functor of the M-type of `P`. `A` is a data-less
possibly infinite tree whereas, for a given `a : A`, `B a` is a valid
path in tree `a` so that `mp α` is made of a tree and a function
from its valid paths to the values it contains -/
def mp : MvPFunctor n where
  A := P.last.M
  B := M.Path P


/-- `n`-ary M-type for `P` -/
def M (α : TypeVec n) : Type _ :=
  P.mp α


                                          /-
                                            n : Nat
                                            P : MvPFunctor.{u} (HAdd.hAdd n 1)
                                            ⊢ MvFunctor P.M
                                          -/
instance mvfunctorM : MvFunctor P.M := by delta M; infer_instance
                                                   /-
                                                     🎉 no goals
                                                   -/


instance inhabitedM {α : TypeVec _} [I : Inhabited P.A] [∀ i : Fin2 n, Inhabited (α i)] :
    Inhabited (P.M α) :=
  @Obj.inhabited _ (mp P) _ (@PFunctor.M.inhabited P.last I) _


/-- construct through corecursion the shape of an M-type
without its contents -/
def M.corecShape {β : Type u} (g₀ : β → P.A) (g₂ : ∀ b : β, P.last.B (g₀ b) → β) :
    β → P.last.M :=
  PFunctor.M.corec fun b => ⟨g₀ b, g₂ b⟩


/-- Proof of type equality as an arrow -/
def castDropB {a a' : P.A} (h : a = a') : P.drop.B a ⟹ P.drop.B a' := fun _i b => Eq.recOn h b


/-- Proof of type equality as a function -/
def castLastB {a a' : P.A} (h : a = a') : P.last.B a → P.last.B a' := fun b => Eq.recOn h b


/-- Using corecursion, construct the contents of an M-type -/
def M.corecContents {α : TypeVec.{u} n}
    {β : Type u}
    (g₀ : β → P.A)
    (g₁ : ∀ b : β, P.drop.B (g₀ b) ⟹ α)
    (g₂ : ∀ b : β, P.last.B (g₀ b) → β)
    (x : _)
    (b : β)
    (h : x = M.corecShape P g₀ g₂ b) :
    M.Path P x ⟹ α
  | _, M.Path.root x a f h' i c =>
    have : a = g₀ b := by
      /-
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        a : P.A
        f : P.last.B a → P.last.M
        h' : Eq x.dest ⟨a, f⟩
        c : P.drop.B a i
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        ⊢ Eq a (g₀ b)
      -/
      rw [h, M.corecShape, PFunctor.M.dest_corec] at h'
      /-
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        a : P.A
        f : P.last.B a → P.last.M
        h' : Eq (P.last.map (PFunctor.M.corec fun b => ⟨g₀ b, g₂ b⟩) ⟨g₀ b, g₂ b⟩) ⟨a, …
        c : P.drop.B a i
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        ⊢ Eq a (g₀ b)
      -/
      cases h'
      /-
        case refl
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        c : P.drop.B (g₀ b) i
        ⊢ Eq (g₀ b) (g₀ b)
      -/
      rfl
      /-
        🎉 no goals
      -/
    g₁ b i (P.castDropB this i c)
  | _, M.Path.child x a f h' j i c =>
    have h₀ : a = g₀ b := by
      /-
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        a : P.A
        f : P.last.B a → P.last.M
        h' : Eq x.dest ⟨a, f⟩
        j : P.last.B a
        c : MvPFunctor.M.Path P (f j) i
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        ⊢ Eq a (g₀ b)
      -/
      rw [h, M.corecShape, PFunctor.M.dest_corec] at h'
      /-
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        a : P.A
        f : P.last.B a → P.last.M
        h' : Eq (P.last.map (PFunctor.M.corec fun b => ⟨g₀ b, g₂ b⟩) ⟨g₀ b, g₂ b⟩) ⟨a, …
        j : P.last.B a
        c : MvPFunctor.M.Path P (f j) i
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        ⊢ Eq a (g₀ b)
      -/
      cases h'
      /-
        case refl
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        j : P.last.B (g₀ b)
        c : MvPFunctor.M.Path P (Function.comp (PFunctor.M.corec fun b => ⟨g₀ b, g₂ b⟩ …
        ⊢ Eq (g₀ b) (g₀ b)
      -/
      rfl
      /-
        🎉 no goals
      -/
    have h₁ : f j = M.corecShape P g₀ g₂ (g₂ b (castLastB P h₀ j)) := by
      /-
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        a : P.A
        f : P.last.B a → P.last.M
        h' : Eq x.dest ⟨a, f⟩
        j : P.last.B a
        c : MvPFunctor.M.Path P (f j) i
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        h₀ : Eq a (g₀ b)
        ⊢ Eq (f j) (MvPFunctor.M.corecShape P g₀ g₂ (g₂ b (P.castLastB h₀ j)))
      -/
      rw [h, M.corecShape, PFunctor.M.dest_corec] at h'
      /-
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        a : P.A
        f : P.last.B a → P.last.M
        h' : Eq (P.last.map (PFunctor.M.corec fun b => ⟨g₀ b, g₂ b⟩) ⟨g₀ b, g₂ b⟩) ⟨a, …
        j : P.last.B a
        c : MvPFunctor.M.Path P (f j) i
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        h₀ : Eq a (g₀ b)
        ⊢ Eq (f j) (MvPFunctor.M.corecShape P g₀ g₂ (g₂ b (P.castLastB h₀ j)))
      -/
      cases h'
      /-
        case refl
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        β : Type u
        g₀ : β → P.A
        g₁ : (b : β) → (P.drop.B (g₀ b)).Arrow α
        g₂ : (b : β) → P.last.B (g₀ b) → β
        x✝ : P.last.M
        b : β
        x : P.last.M
        i : Fin2 n
        h : Eq x (MvPFunctor.M.corecShape P g₀ g₂ b)
        j : P.last.B (g₀ b)
        h₀ : Eq (g₀ b) (g₀ b)
        c : MvPFunctor.M.Path P (Function.comp (PFunctor.M.corec fun b => ⟨g₀ b, g₂ b⟩ …
        ⊢ Eq (Function.comp (PFunctor.M.corec fun b => ⟨g₀ b, g₂ b⟩) (g₂ b) j) (MvPFun …
      -/
      rfl
      /-
        🎉 no goals
      -/
    M.corecContents g₀ g₁ g₂ (f j) (g₂ b (P.castLastB h₀ j)) h₁ i c


/-- Corecursor for M-type of `P` -/
def M.corec' {α : TypeVec n} {β : Type u} (g₀ : β → P.A) (g₁ : ∀ b : β, P.drop.B (g₀ b) ⟹ α)
    (g₂ : ∀ b : β, P.last.B (g₀ b) → β) : β → P.M α := fun b =>
  ⟨M.corecShape P g₀ g₂ b, M.corecContents P g₀ g₁ g₂ _ _ rfl⟩


/-- Corecursor for M-type of `P` -/
def M.corec {α : TypeVec n} {β : Type u} (g : β → P (α.append1 β)) : β → P.M α :=
  M.corec' P (fun b => (g b).fst) (fun b => dropFun (g b).snd) fun b => lastFun (g b).snd


/-- Implementation of destructor for M-type of `P` -/
def M.pathDestLeft {α : TypeVec n} {x : P.last.M} {a : P.A} {f : P.last.B a → P.last.M}
    (h : PFunctor.M.dest x = ⟨a, f⟩) (f' : M.Path P x ⟹ α) : P.drop.B a ⟹ α := fun i c =>
  f' i (M.Path.root x a f h i c)


/-- Implementation of destructor for M-type of `P` -/
def M.pathDestRight {α : TypeVec n} {x : P.last.M} {a : P.A} {f : P.last.B a → P.last.M}
    (h : PFunctor.M.dest x = ⟨a, f⟩) (f' : M.Path P x ⟹ α) :
    ∀ j : P.last.B a, M.Path P (f j) ⟹ α := fun j i c => f' i (M.Path.child x a f h j i c)


/-- Destructor for M-type of `P` -/
def M.dest' {α : TypeVec n} {x : P.last.M} {a : P.A} {f : P.last.B a → P.last.M}
    (h : PFunctor.M.dest x = ⟨a, f⟩) (f' : M.Path P x ⟹ α) : P (α.append1 (P.M α)) :=
  ⟨a, splitFun (M.pathDestLeft P h f') fun x => ⟨f x, M.pathDestRight P h f' x⟩⟩


/-- Destructor for M-types -/
def M.dest {α : TypeVec n} (x : P.M α) : P (α ::: P.M α) :=
  M.dest' P (Sigma.eta <| PFunctor.M.dest x.fst).symm x.snd


/-- Constructor for M-types -/
def M.mk {α : TypeVec n} : P (α.append1 (P.M α)) → P.M α :=
  M.corec _ fun i => appendFun id (M.dest P) <$$> i


theorem M.dest'_eq_dest' {α : TypeVec n} {x : P.last.M} {a₁ : P.A}
    {f₁ : P.last.B a₁ → P.last.M} (h₁ : PFunctor.M.dest x = ⟨a₁, f₁⟩) {a₂ : P.A}
    {f₂ : P.last.B a₂ → P.last.M} (h₂ : PFunctor.M.dest x = ⟨a₂, f₂⟩) (f' : M.Path P x ⟹ α) :
                                            /-
                                              n : Nat
                                              P : MvPFunctor.{u} (HAdd.hAdd n 1)
                                              α : TypeVec.{u} n
                                              x : P.last.M
                                              a₁ : P.A
                                              f₁ : P.last.B a₁ → P.last.M
                                              h₁ : Eq x.dest ⟨a₁, f₁⟩
                                              a₂ : P.A
                                              f₂ : P.last.B a₂ → P.last.M
                                              h₂ : Eq x.dest ⟨a₂, f₂⟩
                                              f' : TypeVec.Arrow (MvPFunctor.M.Path P x) α
                                              ⊢ Eq (MvPFunctor.M.dest' P h₁ f') (MvPFunctor.M.dest' P h₂ f')
                                            -/
    M.dest' P h₁ f' = M.dest' P h₂ f' := by cases h₁.symm.trans h₂; rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem M.dest_eq_dest' {α : TypeVec n} {x : P.last.M} {a : P.A}
    {f : P.last.B a → P.last.M} (h : PFunctor.M.dest x = ⟨a, f⟩) (f' : M.Path P x ⟹ α) :
    M.dest P ⟨x, f'⟩ = M.dest' P h f' :=
  M.dest'_eq_dest' _ _ _ _


theorem M.dest_corec' {α : TypeVec.{u} n} {β : Type u} (g₀ : β → P.A)
    (g₁ : ∀ b : β, P.drop.B (g₀ b) ⟹ α) (g₂ : ∀ b : β, P.last.B (g₀ b) → β) (x : β) :
    M.dest P (M.corec' P g₀ g₁ g₂ x) = ⟨g₀ x, splitFun (g₁ x) (M.corec' P g₀ g₁ g₂ ∘ g₂ x)⟩ :=
  rfl


theorem M.dest_corec {α : TypeVec n} {β : Type u} (g : β → P (α.append1 β)) (x : β) :
    M.dest P (M.corec P g x) = appendFun id (M.corec P g) <$$> g x := by
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    β : Type u
    g : β → ↑P (α.append1 β)
    x : β
    ⊢ Eq (MvPFunctor.M.dest P (MvPFunctor.M.corec P g x)) (MvFunctor.map (TypeVec. …
  -/
  trans
    /-
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      β : Type u
      g : β → ↑P (α.append1 β)
      x : β
      ⊢ Eq (MvPFunctor.M.dest P (MvPFunctor.M.corec P g x)) ?m.11081
    -/
  · apply M.dest_corec'
    /-
      🎉 no goals
    -/
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    β : Type u
    g : β → ↑P (α.append1 β)
    x : β
    ⊢ Eq ⟨(g x).fst, TypeVec.splitFun (TypeVec.dropFun (g x).snd) (Function.comp ( …
  -/
  cases' g x with a f; dsimp
  /-
    case mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    β : Type u
    g : β → ↑P (α.append1 β)
    x : β
    a : P.A
    f : (P.B a).Arrow (α.append1 β)
    ⊢ Eq ⟨a, TypeVec.splitFun (TypeVec.dropFun f) (Function.comp (MvPFunctor.M.cor …
  -/
  rw [MvPFunctor.map_eq]; congr
  /-
    case mk.e_snd
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    β : Type u
    g : β → ↑P (α.append1 β)
    x : β
    a : P.A
    f : (P.B a).Arrow (α.append1 β)
    ⊢ Eq (TypeVec.splitFun (TypeVec.dropFun f) (Function.comp (MvPFunctor.M.corec' …
  -/
  conv_rhs => rw [← split_dropFun_lastFun f, appendFun_comp_splitFun]
  /-
    case mk.e_snd
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    β : Type u
    g : β → ↑P (α.append1 β)
    x : β
    a : P.A
    f : (P.B a).Arrow (α.append1 β)
    ⊢ Eq (TypeVec.splitFun (TypeVec.dropFun f) (Function.comp (MvPFunctor.M.corec' …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem M.bisim_lemma {α : TypeVec n} {a₁ : (mp P).A} {f₁ : (mp P).B a₁ ⟹ α} {a' : P.A}
    {f' : (P.B a').drop ⟹ α} {f₁' : (P.B a').last → M P α}
    (e₁ : M.dest P ⟨a₁, f₁⟩ = ⟨a', splitFun f' f₁'⟩) :
    ∃ (g₁' : _)(e₁' : PFunctor.M.dest a₁ = ⟨a', g₁'⟩),
      f' = M.pathDestLeft P e₁' f₁ ∧
        f₁' = fun x : (last P).B a' => ⟨g₁' x, M.pathDestRight P e₁' f₁ x⟩ := by
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    a' : P.A
    f' : (P.B a').drop.Arrow α
    f₁' : (P.B a').last → P.M α
    e₁ : Eq (MvPFunctor.M.dest P ⟨a₁, f₁⟩) ⟨a', TypeVec.splitFun f' f₁'⟩
    ⊢ Exists fun g₁' => Exists fun e₁' => And (Eq f' (MvPFunctor.M.pathDestLeft P  …
  -/
  generalize ef : @splitFun n _ (append1 α (M P α)) f' f₁' = ff at e₁
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    a' : P.A
    f' : (P.B a').drop.Arrow α
    f₁' : (P.B a').last → P.M α
    ff : (P.B a').Arrow (α.append1 (P.M α))
    ef : Eq (TypeVec.splitFun f' f₁') ff
    e₁ : Eq (MvPFunctor.M.dest P ⟨a₁, f₁⟩) ⟨a', ff⟩
    ⊢ Exists fun g₁' => Exists fun e₁' => And (Eq f' (MvPFunctor.M.pathDestLeft P  …
  -/
  let he₁' := PFunctor.M.dest a₁
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    a' : P.A
    f' : (P.B a').drop.Arrow α
    f₁' : (P.B a').last → P.M α
    ff : (P.B a').Arrow (α.append1 (P.M α))
    ef : Eq (TypeVec.splitFun f' f₁') ff
    e₁ : Eq (MvPFunctor.M.dest P ⟨a₁, f₁⟩) ⟨a', ff⟩
    he₁' : ↑P.last P.last.M := PFunctor.M.dest a₁
    ⊢ Exists fun g₁' => Exists fun e₁' => And (Eq f' (MvPFunctor.M.pathDestLeft P  …
  -/
  rcases e₁' : he₁' with ⟨a₁', g₁'⟩
  /-
    case mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    a' : P.A
    f' : (P.B a').drop.Arrow α
    f₁' : (P.B a').last → P.M α
    ff : (P.B a').Arrow (α.append1 (P.M α))
    ef : Eq (TypeVec.splitFun f' f₁') ff
    e₁ : Eq (MvPFunctor.M.dest P ⟨a₁, f₁⟩) ⟨a', ff⟩
    he₁' : ↑P.last P.last.M := PFunctor.M.dest a₁
    a₁' : P.last.A
    g₁' : P.last.B a₁' → P.last.M
    e₁' : Eq he₁' ⟨a₁', g₁'⟩
    ⊢ Exists fun g₁' => Exists fun e₁' => And (Eq f' (MvPFunctor.M.pathDestLeft P  …
  -/
  rw [M.dest_eq_dest' _ e₁'] at e₁
  /-
    case mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    a' : P.A
    f' : (P.B a').drop.Arrow α
    f₁' : (P.B a').last → P.M α
    ff : (P.B a').Arrow (α.append1 (P.M α))
    ef : Eq (TypeVec.splitFun f' f₁') ff
    he₁' : ↑P.last P.last.M := PFunctor.M.dest a₁
    a₁' : P.last.A
    g₁' : P.last.B a₁' → P.last.M
    e₁' : Eq he₁' ⟨a₁', g₁'⟩
    e₁ : Eq (MvPFunctor.M.dest' P e₁' f₁) ⟨a', ff⟩
    ⊢ Exists fun g₁' => Exists fun e₁' => And (Eq f' (MvPFunctor.M.pathDestLeft P  …
  -/
  cases e₁; exact ⟨_, e₁', splitFun_inj ef⟩
            /-
              🎉 no goals
            -/


theorem M.bisim {α : TypeVec n} (R : P.M α → P.M α → Prop)
    (h :
      ∀ x y,
        R x y →
          ∃ a f f₁ f₂,
            M.dest P x = ⟨a, splitFun f f₁⟩ ∧
              M.dest P y = ⟨a, splitFun f f₂⟩ ∧ ∀ i, R (f₁ i) (f₂ i))
    (x y) (r : R x y) : x = y := by
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
    x y : P.M α
    r : R x y
    ⊢ Eq x y
  -/
  cases' x with a₁ f₁
  /-
    case mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
    y : P.M α
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    r : R ⟨a₁, f₁⟩ y
    ⊢ Eq ⟨a₁, f₁⟩ y
  -/
  cases' y with a₂ f₂
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    a₂ : P.mp.A
    f₂ : (P.mp.B a₂).Arrow α
    r : R ⟨a₁, f₁⟩ ⟨a₂, f₂⟩
    ⊢ Eq ⟨a₁, f₁⟩ ⟨a₂, f₂⟩
  -/
  dsimp [mp] at *
  have : a₁ = a₂ := by
    refine
      PFunctor.M.bisim (fun a₁ a₂ => ∃ x y, R x y ∧ x.1 = a₁ ∧ y.1 = a₂) ?_ _ _
        ⟨⟨a₁, f₁⟩, ⟨a₂, f₂⟩, r, rfl, rfl⟩
    rintro _ _ ⟨⟨a₁, f₁⟩, ⟨a₂, f₂⟩, r, rfl, rfl⟩
    rcases h _ _ r with ⟨a', f', f₁', f₂', e₁, e₂, h'⟩
    rcases M.bisim_lemma P e₁ with ⟨g₁', e₁', rfl, rfl⟩
    rcases M.bisim_lemma P e₂ with ⟨g₂', e₂', _, rfl⟩
    rw [e₁', e₂']
    exact ⟨_, _, _, rfl, rfl, fun b => ⟨_, _, h' b, rfl, rfl⟩⟩
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
    a₁ : P.mp.A
    f₁ : (P.mp.B a₁).Arrow α
    a₂ : P.mp.A
    f₂ : (P.mp.B a₂).Arrow α
    r : R ⟨a₁, f₁⟩ ⟨a₂, f₂⟩
    this : Eq a₁ a₂
    ⊢ Eq ⟨a₁, f₁⟩ ⟨a₂, f₂⟩
  -/
  subst this
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
    a₁ : P.mp.A
    f₁ f₂ : (P.mp.B a₁).Arrow α
    r : R ⟨a₁, f₁⟩ ⟨a₁, f₂⟩
    ⊢ Eq ⟨a₁, f₁⟩ ⟨a₁, f₂⟩
  -/
  congr with (i p)
  /-
    case mk.mk.e_snd.a.h
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
    a₁ : P.mp.A
    f₁ f₂ : (P.mp.B a₁).Arrow α
    r : R ⟨a₁, f₁⟩ ⟨a₁, f₂⟩
    i : Fin2 n
    p : P.mp.B a₁ i
    ⊢ Eq (f₁ i p) (f₂ i p)
  -/
  induction' p with x a f h' i c x a f h' i c p IH <;>
    try
      rcases h _ _ r with ⟨a', f', f₁', f₂', e₁, e₂, h''⟩
      rcases M.bisim_lemma P e₁ with ⟨g₁', e₁', rfl, rfl⟩
      rcases M.bisim_lemma P e₂ with ⟨g₂', e₂', e₃, rfl⟩
      cases h'.symm.trans e₁'
      cases h'.symm.trans e₂'
    /-
      case mk.mk.e_snd.a.h.root.intro.intro.intro.intro.intro.intro.intro.intro.intr …
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
      a₁ : P.mp.A
      i✝ : Fin2 n
      x : P.last.M
      a : P.A
      f : P.last.B a → P.last.M
      h' : Eq x.dest ⟨a, f⟩
      i : Fin2 n
      c : P.drop.B a i
      f₁ f₂ : (P.mp.B x).Arrow α
      r : R ⟨x, f₁⟩ ⟨x, f₂⟩
      e₁' : Eq x.dest ⟨a, f⟩
      e₁ : Eq (MvPFunctor.M.dest P ⟨x, f₁⟩) ⟨a, TypeVec.splitFun (MvPFunctor.M.pathD …
      e₂' : Eq x.dest ⟨a, f⟩
      e₃ : Eq (MvPFunctor.M.pathDestLeft P e₁' f₁) (MvPFunctor.M.pathDestLeft P e₂'  …
      e₂ : Eq (MvPFunctor.M.dest P ⟨x, f₂⟩) ⟨a, TypeVec.splitFun (MvPFunctor.M.pathD …
      h'' : ∀ (i : (P.B a).last), R ((fun x_1 => ⟨f x_1, MvPFunctor.M.pathDestRight  …
      ⊢ Eq (f₁ i (MvPFunctor.M.Path.root x a f h' i c)) (f₂ i (MvPFunctor.M.Path.roo …
    -/
  · exact (congr_fun (congr_fun e₃ i) c : _)
    /-
      🎉 no goals
    -/
    /-
      case mk.mk.e_snd.a.h.child.intro.intro.intro.intro.intro.intro.intro.intro.int …
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ =>  …
      a₁ : P.mp.A
      i✝ : Fin2 n
      x : P.last.M
      a : P.A
      f : P.last.B a → P.last.M
      h' : Eq x.dest ⟨a, f⟩
      i : P.last.B a
      c : Fin2 n
      p : MvPFunctor.M.Path P (f i) c
      IH : ∀ (f₁ f₂ : (P.mp.B (f i)).Arrow α), R ⟨f i, f₁⟩ ⟨f i, f₂⟩ → Eq (f₁ c p) ( …
      f₁ f₂ : (P.mp.B x).Arrow α
      r : R ⟨x, f₁⟩ ⟨x, f₂⟩
      e₁' : Eq x.dest ⟨a, f⟩
      e₁ : Eq (MvPFunctor.M.dest P ⟨x, f₁⟩) ⟨a, TypeVec.splitFun (MvPFunctor.M.pathD …
      e₂' : Eq x.dest ⟨a, f⟩
      e₃ : Eq (MvPFunctor.M.pathDestLeft P e₁' f₁) (MvPFunctor.M.pathDestLeft P e₂'  …
      e₂ : Eq (MvPFunctor.M.dest P ⟨x, f₂⟩) ⟨a, TypeVec.splitFun (MvPFunctor.M.pathD …
      h'' : ∀ (i : (P.B a).last), R ((fun x_1 => ⟨f x_1, MvPFunctor.M.pathDestRight  …
      ⊢ Eq (f₁ c (MvPFunctor.M.Path.child x a f h' i c p)) (f₂ c (MvPFunctor.M.Path. …
    -/
  · exact IH _ _ (h'' _)
    /-
      🎉 no goals
    -/


theorem M.bisim₀ {α : TypeVec n} (R : P.M α → P.M α → Prop) (h₀ : Equivalence R)
    (h : ∀ x y, R x y → (id ::: Quot.mk R) <$$> M.dest _ x = (id ::: Quot.mk R) <$$> M.dest _ y)
    (x y) (r : R x y) : x = y := by
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
    x y : P.M α
    r : R x y
    ⊢ Eq x y
  -/
  apply M.bisim P R _ _ _ r
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
    x y : P.M α
    r : R x y
    ⊢ ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ => Ex …
  -/
  clear r x y
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
    ⊢ ∀ (x y : P.M α), R x y → Exists fun a => Exists fun f => Exists fun f₁ => Ex …
  -/
  introv Hr
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
    x y : P.M α
    Hr : R x y
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq (M …
  -/
  specialize h _ _ Hr
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    Hr : R x y
    h : Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk R)) (MvPFunctor.M …
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq (M …
  -/
  clear Hr

  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    h : Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk R)) (MvPFunctor.M …
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq (M …
  -/
  revert h
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk R)) (MvPFunctor.M.d …
  -/
  rcases M.dest P x with ⟨ax, fx⟩
  /-
    case mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx : (P.B ax).Arrow (α.append1 (P.M α))
    ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk R)) ⟨ax, fx⟩) (MvFu …
  -/
  rcases M.dest P y with ⟨ay, fy⟩
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx : (P.B ax).Arrow (α.append1 (P.M α))
    ay : P.A
    fy : (P.B ay).Arrow (α.append1 (P.M α))
    ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk R)) ⟨ax, fx⟩) (MvFu …
  -/
  intro h

  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx : (P.B ax).Arrow (α.append1 (P.M α))
    ay : P.A
    fy : (P.B ay).Arrow (α.append1 (P.M α))
    h : Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk R)) ⟨ax, fx⟩) (Mv …
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq ⟨a …
  -/
  rw [map_eq, map_eq] at h
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx : (P.B ax).Arrow (α.append1 (P.M α))
    ay : P.A
    fy : (P.B ay).Arrow (α.append1 (P.M α))
    h : Eq ⟨ax, TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx⟩ ⟨ay, T …
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq ⟨a …
  -/
  injection h with h₀ h₁
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀✝ : Equivalence R
    x y : P.M α
    ax : P.A
    fx : (P.B ax).Arrow (α.append1 (P.M α))
    ay : P.A
    fy : (P.B ay).Arrow (α.append1 (P.M α))
    h₀ : Eq ax ay
    h₁ : HEq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx) (TypeVec …
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq ⟨a …
  -/
  subst ay
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    h₁ : HEq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx) (TypeVec …
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq ⟨a …
  -/
  simp? at h₁ says simp only [heq_eq_eq] at h₁
  have Hdrop : dropFun fx = dropFun fy := by
    replace h₁ := congr_arg dropFun h₁
    simpa using h₁
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    h₁ : Eq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx) (TypeVec. …
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    ⊢ Exists fun a => Exists fun f => Exists fun f₁ => Exists fun f₂ => And (Eq ⟨a …
  -/
  exists ax, dropFun fx, lastFun fx, lastFun fy
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    h₁ : Eq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx) (TypeVec. …
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    ⊢ And (Eq ⟨ax, fx⟩ ⟨ax, TypeVec.splitFun (TypeVec.dropFun fx) (TypeVec.lastFun …
  -/
  rw [split_dropFun_lastFun, Hdrop, split_dropFun_lastFun]
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    h₁ : Eq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx) (TypeVec. …
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    ⊢ And (Eq ⟨ax, fx⟩ ⟨ax, fx⟩) (And (Eq ⟨ax, fy⟩ ⟨ax, fy⟩) (∀ (i : (P.B ax).last …
  -/
  simp only [true_and]
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    h₁ : Eq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx) (TypeVec. …
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    ⊢ ∀ (i : (P.B ax).last), R (TypeVec.lastFun fx i) (TypeVec.lastFun fy i)
  -/
  intro i
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    h₁ : Eq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx) (TypeVec. …
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    i : (P.B ax).last
    ⊢ R (TypeVec.lastFun fx i) (TypeVec.lastFun fy i)
  -/
  replace h₁ := congr_fun (congr_fun h₁ Fin2.fz) i
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    i : (P.B ax).last
    h₁ : Eq (TypeVec.comp (TypeVec.appendFun TypeVec.id (Quot.mk R)) fx Fin2.fz i) …
    ⊢ R (TypeVec.lastFun fx i) (TypeVec.lastFun fy i)
  -/
  simp only [TypeVec.comp, appendFun, splitFun] at h₁
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    i : (P.B ax).last
    h₁ : Eq (Quot.mk R (fx Fin2.fz i)) (Quot.mk R (fy Fin2.fz i))
    ⊢ R (TypeVec.lastFun fx i) (TypeVec.lastFun fy i)
  -/
  replace h₁ := Quot.eqvGen_exact h₁
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    i : (P.B ax).last
    h₁ : Relation.EqvGen R (fx Fin2.fz i) (fy Fin2.fz i)
    ⊢ R (TypeVec.lastFun fx i) (TypeVec.lastFun fy i)
  -/
  rw [h₀.eqvGen_iff] at h₁
  /-
    case mk.mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h₀ : Equivalence R
    x y : P.M α
    ax : P.A
    fx fy : (P.B ax).Arrow (α.append1 (P.M α))
    Hdrop : Eq (TypeVec.dropFun fx) (TypeVec.dropFun fy)
    i : (P.B ax).last
    h₁ : R (fx Fin2.fz i) (fy Fin2.fz i)
    ⊢ R (TypeVec.lastFun fx i) (TypeVec.lastFun fy i)
  -/
  exact h₁
  /-
    🎉 no goals
  -/


theorem M.bisim' {α : TypeVec n} (R : P.M α → P.M α → Prop)
    (h : ∀ x y, R x y → (id ::: Quot.mk R) <$$> M.dest _ x = (id ::: Quot.mk R) <$$> M.dest _ y)
    (x y) (r : R x y) : x = y := by
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α : TypeVec.{u} n
    R : P.M α → P.M α → Prop
    h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
    x y : P.M α
    r : R x y
    ⊢ Eq x y
  -/
  have := M.bisim₀ P (Relation.EqvGen R) ?_ ?_
    /-
      case refine_3
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
      x y : P.M α
      r : R x y
      this : ∀ (x y : P.M α), Relation.EqvGen R x y → Eq x y
      ⊢ Eq x y
    -/
  · solve_by_elim [Relation.EqvGen.rel]
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
      x y : P.M α
      r : R x y
      ⊢ Equivalence (Relation.EqvGen R)
    -/
  · apply Relation.EqvGen.is_equivalence
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
      x y : P.M α
      r : R x y
      ⊢ ∀ (x y : P.M α), Relation.EqvGen R x y → Eq (MvFunctor.map (TypeVec.appendFu …
    -/
  · clear r x y
    /-
      case refine_2
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
      ⊢ ∀ (x y : P.M α), Relation.EqvGen R x y → Eq (MvFunctor.map (TypeVec.appendFu …
    -/
    introv Hr
    /-
      case refine_2
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
      x y : P.M α
      Hr : Relation.EqvGen R x y
      ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk (Relation.EqvGen R) …
    -/
    have : ∀ x y, R x y → Relation.EqvGen R x y := @Relation.EqvGen.rel _ R
    /-
      case refine_2
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
      x y : P.M α
      Hr : Relation.EqvGen R x y
      this : ∀ (x y : P.M α), R x y → Relation.EqvGen R x y
      ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk (Relation.EqvGen R) …
    -/
    induction Hr
      /-
        case refine_2.rel
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        R : P.M α → P.M α → Prop
        h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
        x y : P.M α
        this : ∀ (x y : P.M α), R x y → Relation.EqvGen R x y
        x✝ y✝ : P.M α
        a✝ : R x✝ y✝
        ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk (Relation.EqvGen R) …
      -/
    · rw [← Quot.factor_mk_eq R (Relation.EqvGen R) this]
      /-
        case refine_2.rel
        n : Nat
        P : MvPFunctor.{u} (HAdd.hAdd n 1)
        α : TypeVec.{u} n
        R : P.M α → P.M α → Prop
        h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
        x y : P.M α
        this : ∀ (x y : P.M α), R x y → Relation.EqvGen R x y
        x✝ y✝ : P.M α
        a✝ : R x✝ y✝
        ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Function.comp (Quot.factor  …
      -/
      rwa [appendFun_comp_id, ← MvFunctor.map_map, ← MvFunctor.map_map, h]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.refl
      n : Nat
      P : MvPFunctor.{u} (HAdd.hAdd n 1)
      α : TypeVec.{u} n
      R : P.M α → P.M α → Prop
      h : ∀ (x y : P.M α), R x y → Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id ( …
      x y : P.M α
      this : ∀ (x y : P.M α), R x y → Relation.EqvGen R x y
      x✝ : P.M α
      ⊢ Eq (MvFunctor.map (TypeVec.appendFun TypeVec.id (Quot.mk (Relation.EqvGen R) …
    -/
    all_goals aesop
    /-
      🎉 no goals
    -/


theorem M.dest_map {α β : TypeVec n} (g : α ⟹ β) (x : P.M α) :
    M.dest P (g <$$> x) = (appendFun g fun x => g <$$> x) <$$> M.dest P x := by
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α β : TypeVec.{u} n
    g : α.Arrow β
    x : P.M α
    ⊢ Eq (MvPFunctor.M.dest P (MvFunctor.map g x)) (MvFunctor.map (TypeVec.appendF …
  -/
  cases' x with a f
  /-
    case mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α β : TypeVec.{u} n
    g : α.Arrow β
    a : P.mp.A
    f : (P.mp.B a).Arrow α
    ⊢ Eq (MvPFunctor.M.dest P (MvFunctor.map g ⟨a, f⟩)) (MvFunctor.map (TypeVec.ap …
  -/
  rw [map_eq]
  conv =>
    rhs
    rw [M.dest, M.dest', map_eq, appendFun_comp_splitFun]
  /-
    case mk
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α β : TypeVec.{u} n
    g : α.Arrow β
    a : P.mp.A
    f : (P.mp.B a).Arrow α
    ⊢ Eq (MvPFunctor.M.dest P ⟨a, TypeVec.comp g f⟩) ⟨(PFunctor.M.dest ⟨a, f⟩.fst) …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem M.map_dest {α β : TypeVec n} (g : (α ::: P.M α) ⟹ (β ::: P.M β)) (x : P.M α)
    (h : ∀ x : P.M α, lastFun g x = (dropFun g <$$> x : P.M β)) :
    g <$$> M.dest P x = M.dest P (dropFun g <$$> x) := by
  /-
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α β : TypeVec.{u} n
    g : (α.append1 (P.M α)).Arrow (β.append1 (P.M β))
    x : P.M α
    h : ∀ (x : P.M α), Eq (TypeVec.lastFun g x) (MvFunctor.map (TypeVec.dropFun g) …
    ⊢ Eq (MvFunctor.map g (MvPFunctor.M.dest P x)) (MvPFunctor.M.dest P (MvFunctor …
  -/
  rw [M.dest_map]; congr
  /-
    case e_a
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α β : TypeVec.{u} n
    g : (α.append1 (P.M α)).Arrow (β.append1 (P.M β))
    x : P.M α
    h : ∀ (x : P.M α), Eq (TypeVec.lastFun g x) (MvFunctor.map (TypeVec.dropFun g) …
    ⊢ Eq g (TypeVec.appendFun (TypeVec.dropFun g) fun x => MvFunctor.map (TypeVec. …
  -/
  apply eq_of_drop_last_eq (by simp)
  /-
    case e_a
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α β : TypeVec.{u} n
    g : (α.append1 (P.M α)).Arrow (β.append1 (P.M β))
    x : P.M α
    h : ∀ (x : P.M α), Eq (TypeVec.lastFun g x) (MvFunctor.map (TypeVec.dropFun g) …
    ⊢ Eq (TypeVec.lastFun g) (TypeVec.lastFun (TypeVec.appendFun (TypeVec.dropFun  …
  -/
  simp only [lastFun_appendFun]
  /-
    case e_a
    n : Nat
    P : MvPFunctor.{u} (HAdd.hAdd n 1)
    α β : TypeVec.{u} n
    g : (α.append1 (P.M α)).Arrow (β.append1 (P.M β))
    x : P.M α
    h : ∀ (x : P.M α), Eq (TypeVec.lastFun g x) (MvFunctor.map (TypeVec.dropFun g) …
    ⊢ Eq (TypeVec.lastFun g) fun x => MvFunctor.map (TypeVec.dropFun g) x
  -/
  ext1; apply h
        /-
          🎉 no goals
        -/


