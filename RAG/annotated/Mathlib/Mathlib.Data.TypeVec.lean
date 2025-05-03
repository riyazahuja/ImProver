/-- n-tuples of types, as a category -/
@[pp_with_univ]
def TypeVec (n : ℕ) :=
  Fin2 n → Type*


instance {n} : Inhabited (TypeVec.{u} n) :=
  ⟨fun _ => PUnit⟩


/-- arrow in the category of `TypeVec` -/
def Arrow (α β : TypeVec n) :=
  ∀ i : Fin2 n, α i → β i


@[inherit_doc] scoped[MvFunctor] infixl:40 " ⟹ " => TypeVec.Arrow

/-- Extensionality for arrows -/
@[ext]
theorem Arrow.ext {α β : TypeVec n} (f g : α ⟹ β) :
    (∀ i, f i = g i) → f = g := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    β : TypeVec.{u_2} n
    f g : α.Arrow β
    ⊢ (∀ (i : Fin2 n), Eq (f i) (g i)) → Eq f g
  -/
  intro h; funext i; apply h
                     /-
                       🎉 no goals
                     -/


instance Arrow.inhabited (α β : TypeVec n) [∀ i, Inhabited (β i)] : Inhabited (α ⟹ β) :=
  ⟨fun _ _ => default⟩


/-- identity of arrow composition -/
def id {α : TypeVec n} : α ⟹ α := fun _ x => x


/-- arrow composition in the category of `TypeVec` -/
def comp {α β γ : TypeVec n} (g : β ⟹ γ) (f : α ⟹ β) : α ⟹ γ := fun i x => g i (f i x)


@[inherit_doc] scoped[MvFunctor] infixr:80 " ⊚ " => TypeVec.comp -- type as \oo


@[simp]
theorem id_comp {α β : TypeVec n} (f : α ⟹ β) : id ⊚ f = f :=
  rfl


@[simp]
theorem comp_id {α β : TypeVec n} (f : α ⟹ β) : f ⊚ id = f :=
  rfl


theorem comp_assoc {α β γ δ : TypeVec n} (h : γ ⟹ δ) (g : β ⟹ γ) (f : α ⟹ β) :
    (h ⊚ g) ⊚ f = h ⊚ g ⊚ f :=
  rfl


/-- Support for extending a `TypeVec` by one element. -/
def append1 (α : TypeVec n) (β : Type*) : TypeVec (n + 1)
  | Fin2.fs i => α i
  | Fin2.fz => β


@[inherit_doc] infixl:67 " ::: " => append1


/-- retain only a `n-length` prefix of the argument -/
def drop (α : TypeVec.{u} (n + 1)) : TypeVec n := fun i => α i.fs


/-- take the last value of a `(n+1)-length` vector -/
def last (α : TypeVec.{u} (n + 1)) : Type _ :=
  α Fin2.fz


instance last.inhabited (α : TypeVec (n + 1)) [Inhabited (α Fin2.fz)] : Inhabited (last α) :=
  ⟨show α Fin2.fz from default⟩


theorem drop_append1 {α : TypeVec n} {β : Type*} {i : Fin2 n} : drop (append1 α β) i = α i :=
  rfl


theorem drop_append1' {α : TypeVec n} {β : Type*} : drop (append1 α β) = α :=
  funext fun _ => drop_append1


theorem last_append1 {α : TypeVec n} {β : Type*} : last (append1 α β) = β :=
  rfl


@[simp]
theorem append1_drop_last (α : TypeVec (n + 1)) : append1 (drop α) (last α) = α :=
                     /-
                       n : Nat
                       α : TypeVec.{u_1} (HAdd.hAdd n 1)
                       i : Fin2 (HAdd.hAdd n 1)
                       ⊢ Eq (α.drop.append1 α.last i) (α i)
                     -/
                                 /-
                                   🎉 no goals
                                 -/
  funext fun i => by cases i <;> rfl
                                 /-
                                   🎉 no goals
                                 -/


/-- cases on `(n+1)-length` vectors -/
@[elab_as_elim]
def append1Cases {C : TypeVec (n + 1) → Sort u} (H : ∀ α β, C (append1 α β)) (γ) : C γ := by
  /-
    n : Nat
    C : TypeVec.{?u.7275} (HAdd.hAdd n 1) → Sort u
    H : (α : TypeVec.{?u.7275} n) → (β : Type ?u.7275) → C (α.append1 β)
    γ : TypeVec.{?u.7275} (HAdd.hAdd n 1)
    ⊢ C γ
  -/
  rw [← @append1_drop_last _ γ]; apply H
                                 /-
                                   🎉 no goals
                                 -/


@[simp]
theorem append1_cases_append1 {C : TypeVec (n + 1) → Sort u} (H : ∀ α β, C (append1 α β)) (α β) :
    @append1Cases _ C H (append1 α β) = H α β :=
  rfl


/-- append an arrow and a function for arbitrary source and target type vectors -/
def splitFun {α α' : TypeVec (n + 1)} (f : drop α ⟹ drop α') (g : last α → last α') : α ⟹ α'
  | Fin2.fs i => f i
  | Fin2.fz => g


/-- append an arrow and a function as well as their respective source and target types / typevecs -/
def appendFun {α α' : TypeVec n} {β β' : Type*} (f : α ⟹ α') (g : β → β') :
    append1 α β ⟹ append1 α' β' :=
  splitFun f g


@[inherit_doc] infixl:0 " ::: " => appendFun


/-- split off the prefix of an arrow -/
def dropFun {α β : TypeVec (n + 1)} (f : α ⟹ β) : drop α ⟹ drop β := fun i => f i.fs


/-- split off the last function of an arrow -/
def lastFun {α β : TypeVec (n + 1)} (f : α ⟹ β) : last α → last β :=
  f Fin2.fz

-- Porting note: Lean wasn't able to infer the motive in term mode

/-- arrow in the category of `0-length` vectors -/
                                                                  /-
                                                                    n : Nat
                                                                    α : TypeVec.{?u.10141} 0
                                                                    β : TypeVec.{?u.10155} 0
                                                                    i : Fin2 0
                                                                    ⊢ α i → β i
                                                                  -/
def nilFun {α : TypeVec 0} {β : TypeVec 0} : α ⟹ β := fun i => by apply Fin2.elim0 i
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem eq_of_drop_last_eq {α β : TypeVec (n + 1)} {f g : α ⟹ β} (h₀ : dropFun f = dropFun g)
    (h₁ : lastFun f = lastFun g) : f = g := by
  -- Porting note: FIXME: congr_fun h₀ <;> ext1 ⟨⟩ <;> apply_assumption
  /-
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    β : TypeVec.{u_2} (HAdd.hAdd n 1)
    f g : α.Arrow β
    h₀ : Eq (TypeVec.dropFun f) (TypeVec.dropFun g)
    h₁ : Eq (TypeVec.lastFun f) (TypeVec.lastFun g)
    ⊢ Eq f g
  -/
  refine funext (fun x => ?_)
  /-
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    β : TypeVec.{u_2} (HAdd.hAdd n 1)
    f g : α.Arrow β
    h₀ : Eq (TypeVec.dropFun f) (TypeVec.dropFun g)
    h₁ : Eq (TypeVec.lastFun f) (TypeVec.lastFun g)
    x : Fin2 (HAdd.hAdd n 1)
    ⊢ Eq (f x) (g x)
  -/
  cases x
    /-
      case fz
      n : Nat
      α : TypeVec.{u_1} (HAdd.hAdd n 1)
      β : TypeVec.{u_2} (HAdd.hAdd n 1)
      f g : α.Arrow β
      h₀ : Eq (TypeVec.dropFun f) (TypeVec.dropFun g)
      h₁ : Eq (TypeVec.lastFun f) (TypeVec.lastFun g)
      ⊢ Eq (f Fin2.fz) (g Fin2.fz)
    -/
  · apply h₁
    /-
      🎉 no goals
    -/
    /-
      case fs
      n : Nat
      α : TypeVec.{u_1} (HAdd.hAdd n 1)
      β : TypeVec.{u_2} (HAdd.hAdd n 1)
      f g : α.Arrow β
      h₀ : Eq (TypeVec.dropFun f) (TypeVec.dropFun g)
      h₁ : Eq (TypeVec.lastFun f) (TypeVec.lastFun g)
      a✝ : Fin2 n
      ⊢ Eq (f a✝.fs) (g a✝.fs)
    -/
  · apply congr_fun h₀
    /-
      🎉 no goals
    -/


@[simp]
theorem dropFun_splitFun {α α' : TypeVec (n + 1)} (f : drop α ⟹ drop α') (g : last α → last α') :
    dropFun (splitFun f g) = f :=
  rfl


/-- turn an equality into an arrow -/
def Arrow.mp {α β : TypeVec n} (h : α = β) : α ⟹ β
  | _ => Eq.mp (congr_fun h _)


/-- turn an equality into an arrow, with reverse direction -/
def Arrow.mpr {α β : TypeVec n} (h : α = β) : β ⟹ α
  | _ => Eq.mpr (congr_fun h _)


/-- decompose a vector into its prefix appended with its last element -/
def toAppend1DropLast {α : TypeVec (n + 1)} : α ⟹ (drop α ::: last α) :=
  Arrow.mpr (append1_drop_last _)


/-- stitch two bits of a vector back together -/
def fromAppend1DropLast {α : TypeVec (n + 1)} : (drop α ::: last α) ⟹ α :=
  Arrow.mp (append1_drop_last _)


@[simp]
theorem lastFun_splitFun {α α' : TypeVec (n + 1)} (f : drop α ⟹ drop α') (g : last α → last α') :
    lastFun (splitFun f g) = g :=
  rfl


@[simp]
theorem dropFun_appendFun {α α' : TypeVec n} {β β' : Type*} (f : α ⟹ α') (g : β → β') :
    dropFun (f ::: g) = f :=
  rfl


@[simp]
theorem lastFun_appendFun {α α' : TypeVec n} {β β' : Type*} (f : α ⟹ α') (g : β → β') :
    lastFun (f ::: g) = g :=
  rfl


theorem split_dropFun_lastFun {α α' : TypeVec (n + 1)} (f : α ⟹ α') :
    splitFun (dropFun f) (lastFun f) = f :=
  eq_of_drop_last_eq rfl rfl


theorem splitFun_inj {α α' : TypeVec (n + 1)} {f f' : drop α ⟹ drop α'} {g g' : last α → last α'}
    (H : splitFun f g = splitFun f' g') : f = f' ∧ g = g' := by
  /-
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    α' : TypeVec.{u_2} (HAdd.hAdd n 1)
    f f' : α.drop.Arrow α'.drop
    g g' : α.last → α'.last
    H : Eq (TypeVec.splitFun f g) (TypeVec.splitFun f' g')
    ⊢ And (Eq f f') (Eq g g')
  -/
  rw [← dropFun_splitFun f g, H, ← lastFun_splitFun f g, H]; simp
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem appendFun_inj {α α' : TypeVec n} {β β' : Type*} {f f' : α ⟹ α'} {g g' : β → β'} :
    (f ::: g : (α ::: β) ⟹ _) = (f' ::: g' : (α ::: β) ⟹ _)
    → f = f' ∧ g = g' :=
  splitFun_inj


theorem splitFun_comp {α₀ α₁ α₂ : TypeVec (n + 1)} (f₀ : drop α₀ ⟹ drop α₁)
    (f₁ : drop α₁ ⟹ drop α₂) (g₀ : last α₀ → last α₁) (g₁ : last α₁ → last α₂) :
    splitFun (f₁ ⊚ f₀) (g₁ ∘ g₀) = splitFun f₁ g₁ ⊚ splitFun f₀ g₀ :=
  eq_of_drop_last_eq rfl rfl


theorem appendFun_comp_splitFun {α γ : TypeVec n} {β δ : Type*} {ε : TypeVec (n + 1)}
    (f₀ : drop ε ⟹ α) (f₁ : α ⟹ γ) (g₀ : last ε → β) (g₁ : β → δ) :
    appendFun f₁ g₁ ⊚ splitFun f₀ g₀ = splitFun (α' := γ.append1 δ) (f₁ ⊚ f₀) (g₁ ∘ g₀) :=
  (splitFun_comp _ _ _ _).symm


theorem appendFun_comp  {α₀ α₁ α₂ : TypeVec n}
    {β₀ β₁ β₂ : Type*}
    (f₀ : α₀ ⟹ α₁) (f₁ : α₁ ⟹ α₂)
    (g₀ : β₀ → β₁) (g₁ : β₁ → β₂) :
    (f₁ ⊚ f₀ ::: g₁ ∘ g₀) = (f₁ ::: g₁) ⊚ (f₀ ::: g₀) :=
  eq_of_drop_last_eq rfl rfl


theorem appendFun_comp' {α₀ α₁ α₂ : TypeVec n} {β₀ β₁ β₂ : Type*}
    (f₀ : α₀ ⟹ α₁) (f₁ : α₁ ⟹ α₂) (g₀ : β₀ → β₁) (g₁ : β₁ → β₂) :
    (f₁ ::: g₁) ⊚ (f₀ ::: g₀) = (f₁ ⊚ f₀ ::: g₁ ∘ g₀) :=
  eq_of_drop_last_eq rfl rfl


theorem nilFun_comp {α₀ : TypeVec 0} (f₀ : α₀ ⟹ Fin2.elim0) : nilFun ⊚ f₀ = f₀ :=
                     /-
                       α₀ : TypeVec.{u_1} 0
                       f₀ : α₀.Arrow Fin2.elim0
                       x : Fin2 0
                       ⊢ Eq (TypeVec.comp TypeVec.nilFun f₀ x) (f₀ x)
                     -/
  funext fun x => by apply Fin2.elim0 x -- Porting note: `by apply` is necessary?
                     /-
                       🎉 no goals
                     -/


theorem appendFun_comp_id {α : TypeVec n} {β₀ β₁ β₂ : Type u} (g₀ : β₀ → β₁) (g₁ : β₁ → β₂) :
    (@id _ α ::: g₁ ∘ g₀) = (id ::: g₁) ⊚ (id ::: g₀) :=
  eq_of_drop_last_eq rfl rfl


@[simp]
theorem dropFun_comp {α₀ α₁ α₂ : TypeVec (n + 1)} (f₀ : α₀ ⟹ α₁) (f₁ : α₁ ⟹ α₂) :
    dropFun (f₁ ⊚ f₀) = dropFun f₁ ⊚ dropFun f₀ :=
  rfl


@[simp]
theorem lastFun_comp {α₀ α₁ α₂ : TypeVec (n + 1)} (f₀ : α₀ ⟹ α₁) (f₁ : α₁ ⟹ α₂) :
    lastFun (f₁ ⊚ f₀) = lastFun f₁ ∘ lastFun f₀ :=
  rfl


theorem appendFun_aux {α α' : TypeVec n} {β β' : Type*} (f : (α ::: β) ⟹ (α' ::: β')) :
    (dropFun f ::: lastFun f) = f :=
  eq_of_drop_last_eq rfl rfl


theorem appendFun_id_id {α : TypeVec n} {β : Type*} :
    (@TypeVec.id n α ::: @_root_.id β) = TypeVec.id :=
  eq_of_drop_last_eq rfl rfl


instance subsingleton0 : Subsingleton (TypeVec 0) :=
                                 /-
                                   n : Nat
                                   a✝ b : TypeVec.{u_1} 0
                                   a : Fin2 0
                                   ⊢ Eq (a✝ a) (b a)
                                 -/
  ⟨fun a b => funext fun a => by apply Fin2.elim0 a⟩ -- Porting note: `by apply` necessary?
                                 /-
                                   🎉 no goals
                                 -/

-- Porting note: `simp` attribute `TypeVec` moved to file `Tactic/Attr/Register.lean`


/-- cases distinction for 0-length type vector -/
protected def casesNil {β : TypeVec 0 → Sort*} (f : β Fin2.elim0) : ∀ v, β v :=
                    /-
                      n : Nat
                      β : TypeVec.{?u.22842} 0 → Sort u_1
                      f : β Fin2.elim0
                      v : TypeVec.{?u.22842} 0
                      ⊢ Eq (β Fin2.elim0) (β v)
                    -/
  fun v => cast (by congr; funext i; cases i) f
                                     /-
                                       🎉 no goals
                                     -/


/-- cases distinction for (n+1)-length type vector -/
protected def casesCons (n : ℕ) {β : TypeVec (n + 1) → Sort*}
    (f : ∀ (t) (v : TypeVec n), β (v ::: t)) :
    ∀ v, β v :=
                                      /-
                                        n✝ n : Nat
                                        β : TypeVec.{?u.23592} (HAdd.hAdd n 1) → Sort u_1
                                        f : (t : Type ?u.23592) → (v : TypeVec.{?u.23592} n) → β (v.append1 t)
                                        v : TypeVec.{?u.23592} (HAdd.hAdd n 1)
                                        ⊢ Eq (β (v.drop.append1 v.last)) (β v)
                                      -/
  fun v : TypeVec (n + 1) => cast (by simp) (f v.last v.drop)
                                      /-
                                        🎉 no goals
                                      -/


protected theorem casesNil_append1 {β : TypeVec 0 → Sort*} (f : β Fin2.elim0) :
    TypeVec.casesNil f Fin2.elim0 = f :=
  rfl


protected theorem casesCons_append1 (n : ℕ) {β : TypeVec (n + 1) → Sort*}
    (f : ∀ (t) (v : TypeVec n), β (v ::: t)) (v : TypeVec n) (α) :
    TypeVec.casesCons n f (v ::: α) = f α v :=
  rfl


/-- cases distinction for an arrow in the category of 0-length type vectors -/
def typevecCasesNil₃ {β : ∀ v v' : TypeVec 0, v ⟹ v' → Sort*}
    (f : β Fin2.elim0 Fin2.elim0 nilFun) :
    ∀ v v' fs, β v v' fs := fun v v' fs => by
  /-
    n : Nat
    β : (v : TypeVec.{?u.24344} 0) → (v' : TypeVec.{?u.24358} 0) → v.Arrow v' → So …
    f : β Fin2.elim0 Fin2.elim0 TypeVec.nilFun
    v : TypeVec.{?u.24344} 0
    v' : TypeVec.{?u.24358} 0
    fs : v.Arrow v'
    ⊢ β v v' fs
  -/
  refine cast ?_ f
  /-
    n : Nat
    β : (v : TypeVec.{?u.24344} 0) → (v' : TypeVec.{?u.24358} 0) → v.Arrow v' → So …
    f : β Fin2.elim0 Fin2.elim0 TypeVec.nilFun
    v : TypeVec.{?u.24344} 0
    v' : TypeVec.{?u.24358} 0
    fs : v.Arrow v'
    ⊢ Eq (β Fin2.elim0 Fin2.elim0 TypeVec.nilFun) (β v v' fs)
  -/
  have eq₁ : v = Fin2.elim0 := by funext i; contradiction
  /-
    n : Nat
    β : (v : TypeVec.{?u.24344} 0) → (v' : TypeVec.{?u.24358} 0) → v.Arrow v' → So …
    f : β Fin2.elim0 Fin2.elim0 TypeVec.nilFun
    v : TypeVec.{?u.24344} 0
    v' : TypeVec.{?u.24358} 0
    fs : v.Arrow v'
    eq₁ : Eq v Fin2.elim0
    ⊢ Eq (β Fin2.elim0 Fin2.elim0 TypeVec.nilFun) (β v v' fs)
  -/
  have eq₂ : v' = Fin2.elim0 := by funext i; contradiction
  /-
    n : Nat
    β : (v : TypeVec.{?u.24344} 0) → (v' : TypeVec.{?u.24358} 0) → v.Arrow v' → So …
    f : β Fin2.elim0 Fin2.elim0 TypeVec.nilFun
    v : TypeVec.{?u.24344} 0
    v' : TypeVec.{?u.24358} 0
    fs : v.Arrow v'
    eq₁ : Eq v Fin2.elim0
    eq₂ : Eq v' Fin2.elim0
    ⊢ Eq (β Fin2.elim0 Fin2.elim0 TypeVec.nilFun) (β v v' fs)
  -/
  have eq₃ : fs = nilFun := by funext i; contradiction
  /-
    n : Nat
    β : (v : TypeVec.{?u.24344} 0) → (v' : TypeVec.{?u.24358} 0) → v.Arrow v' → So …
    f : β Fin2.elim0 Fin2.elim0 TypeVec.nilFun
    v : TypeVec.{?u.24344} 0
    v' : TypeVec.{?u.24358} 0
    fs : v.Arrow v'
    eq₁ : Eq v Fin2.elim0
    eq₂ : Eq v' Fin2.elim0
    eq₃ : Eq fs TypeVec.nilFun
    ⊢ Eq (β Fin2.elim0 Fin2.elim0 TypeVec.nilFun) (β v v' fs)
  -/
  cases eq₁; cases eq₂; cases eq₃; rfl
                                   /-
                                     🎉 no goals
                                   -/


/-- cases distinction for an arrow in the category of (n+1)-length type vectors -/
def typevecCasesCons₃ (n : ℕ) {β : ∀ v v' : TypeVec (n + 1), v ⟹ v' → Sort*}
    (F : ∀ (t t') (f : t → t') (v v' : TypeVec n) (fs : v ⟹ v'),
    β (v ::: t) (v' ::: t') (fs ::: f)) :
    ∀ v v' fs, β v v' fs := by
  /-
    n✝ n : Nat
    β : (v : TypeVec.{?u.26115} (HAdd.hAdd n 1)) → (v' : TypeVec.{?u.26118} (HAdd. …
    F : (t : Type ?u.26115) → (t' : Type ?u.26118) → (f : t → t') → (v : TypeVec.{ …
    ⊢ (v : TypeVec.{?u.26115} (HAdd.hAdd n 1)) → (v' : TypeVec.{?u.26118} (HAdd.hA …
  -/
  intro v v'
  /-
    n✝ n : Nat
    β : (v : TypeVec.{?u.26115} (HAdd.hAdd n 1)) → (v' : TypeVec.{?u.26118} (HAdd. …
    F : (t : Type ?u.26115) → (t' : Type ?u.26118) → (f : t → t') → (v : TypeVec.{ …
    v : TypeVec.{?u.26115} (HAdd.hAdd n 1)
    v' : TypeVec.{?u.26118} (HAdd.hAdd n 1)
    ⊢ (fs : v.Arrow v') → β v v' fs
  -/
  rw [← append1_drop_last v, ← append1_drop_last v']
  /-
    n✝ n : Nat
    β : (v : TypeVec.{?u.26115} (HAdd.hAdd n 1)) → (v' : TypeVec.{?u.26118} (HAdd. …
    F : (t : Type ?u.26115) → (t' : Type ?u.26118) → (f : t → t') → (v : TypeVec.{ …
    v : TypeVec.{?u.26115} (HAdd.hAdd n 1)
    v' : TypeVec.{?u.26118} (HAdd.hAdd n 1)
    ⊢ (fs : (v.drop.append1 v.last).Arrow (v'.drop.append1 v'.last)) → β (v.drop.a …
  -/
  intro fs
  /-
    n✝ n : Nat
    β : (v : TypeVec.{?u.26115} (HAdd.hAdd n 1)) → (v' : TypeVec.{?u.26118} (HAdd. …
    F : (t : Type ?u.26115) → (t' : Type ?u.26118) → (f : t → t') → (v : TypeVec.{ …
    v : TypeVec.{?u.26115} (HAdd.hAdd n 1)
    v' : TypeVec.{?u.26118} (HAdd.hAdd n 1)
    fs : (v.drop.append1 v.last).Arrow (v'.drop.append1 v'.last)
    ⊢ β (v.drop.append1 v.last) (v'.drop.append1 v'.last) fs
  -/
  rw [← split_dropFun_lastFun fs]
  /-
    n✝ n : Nat
    β : (v : TypeVec.{?u.26115} (HAdd.hAdd n 1)) → (v' : TypeVec.{?u.26118} (HAdd. …
    F : (t : Type ?u.26115) → (t' : Type ?u.26118) → (f : t → t') → (v : TypeVec.{ …
    v : TypeVec.{?u.26115} (HAdd.hAdd n 1)
    v' : TypeVec.{?u.26118} (HAdd.hAdd n 1)
    fs : (v.drop.append1 v.last).Arrow (v'.drop.append1 v'.last)
    ⊢ β (v.drop.append1 v.last) (v'.drop.append1 v'.last) (TypeVec.splitFun (TypeV …
  -/
  apply F
  /-
    🎉 no goals
  -/


/-- specialized cases distinction for an arrow in the category of 0-length type vectors -/
def typevecCasesNil₂ {β : Fin2.elim0 ⟹ Fin2.elim0 → Sort*} (f : β nilFun) : ∀ f, β f := by
  /-
    n : Nat
    β : TypeVec.Arrow Fin2.elim0 Fin2.elim0 → Sort u_1
    f : β TypeVec.nilFun
    ⊢ (f : TypeVec.Arrow Fin2.elim0 Fin2.elim0) → β f
  -/
  intro g
  /-
    n : Nat
    β : TypeVec.Arrow Fin2.elim0 Fin2.elim0 → Sort u_1
    f : β TypeVec.nilFun
    g : TypeVec.Arrow Fin2.elim0 Fin2.elim0
    ⊢ β g
  -/
  suffices g = nilFun by rwa [this]
  /-
    n : Nat
    β : TypeVec.Arrow Fin2.elim0 Fin2.elim0 → Sort u_1
    f : β TypeVec.nilFun
    g : TypeVec.Arrow Fin2.elim0 Fin2.elim0
    ⊢ Eq g TypeVec.nilFun
  -/
  ext ⟨⟩
  /-
    🎉 no goals
  -/


/-- specialized cases distinction for an arrow in the category of (n+1)-length type vectors -/
def typevecCasesCons₂ (n : ℕ) (t t' : Type*) (v v' : TypeVec n)
    {β : (v ::: t) ⟹ (v' ::: t') → Sort*}
    (F : ∀ (f : t → t') (fs : v ⟹ v'), β (fs ::: f)) : ∀ fs, β fs := by
  /-
    n✝ n : Nat
    t : Type u_1
    t' : Type u_2
    v : TypeVec.{u_1} n
    v' : TypeVec.{u_2} n
    β : (v.append1 t).Arrow (v'.append1 t') → Sort u_3
    F : (f : t → t') → (fs : v.Arrow v') → β (TypeVec.appendFun fs f)
    ⊢ (fs : (v.append1 t).Arrow (v'.append1 t')) → β fs
  -/
  intro fs
  /-
    n✝ n : Nat
    t : Type u_1
    t' : Type u_2
    v : TypeVec.{u_1} n
    v' : TypeVec.{u_2} n
    β : (v.append1 t).Arrow (v'.append1 t') → Sort u_3
    F : (f : t → t') → (fs : v.Arrow v') → β (TypeVec.appendFun fs f)
    fs : (v.append1 t).Arrow (v'.append1 t')
    ⊢ β fs
  -/
  rw [← split_dropFun_lastFun fs]
  /-
    n✝ n : Nat
    t : Type u_1
    t' : Type u_2
    v : TypeVec.{u_1} n
    v' : TypeVec.{u_2} n
    β : (v.append1 t).Arrow (v'.append1 t') → Sort u_3
    F : (f : t → t') → (fs : v.Arrow v') → β (TypeVec.appendFun fs f)
    fs : (v.append1 t).Arrow (v'.append1 t')
    ⊢ β (TypeVec.splitFun (TypeVec.dropFun fs) (TypeVec.lastFun fs))
  -/
  apply F
  /-
    🎉 no goals
  -/



theorem typevecCasesNil₂_appendFun {β : Fin2.elim0 ⟹ Fin2.elim0 → Sort*} (f : β nilFun) :
    typevecCasesNil₂ f nilFun = f :=
  rfl


theorem typevecCasesCons₂_appendFun (n : ℕ) (t t' : Type*) (v v' : TypeVec n)
    {β : (v ::: t) ⟹ (v' ::: t') → Sort*}
    (F : ∀ (f : t → t') (fs : v ⟹ v'), β (fs ::: f))
    (f fs) :
    typevecCasesCons₂ n t t' v v' F (fs ::: f) = F f fs :=
  rfl

-- for lifting predicates and relations

/-- `PredLast α p x` predicates `p` of the last element of `x : α.append1 β`. -/
def PredLast (α : TypeVec n) {β : Type*} (p : β → Prop) : ∀ ⦃i⦄, (α.append1 β) i → Prop
  | Fin2.fs _ => fun _ => True
  | Fin2.fz => p


/-- `RelLast α r x y` says that `p` the last elements of `x y : α.append1 β` are related by `r` and
all the other elements are equal. -/
def RelLast (α : TypeVec n) {β γ : Type u} (r : β → γ → Prop) :
    ∀ ⦃i⦄, (α.append1 β) i → (α.append1 γ) i → Prop
  | Fin2.fs _ => Eq
  | Fin2.fz => r


/-- `repeat n t` is a `n-length` type vector that contains `n` occurrences of `t` -/
def «repeat» : ∀ (n : ℕ), Sort _ → TypeVec n
  | 0, _ => Fin2.elim0
  | Nat.succ i, t => append1 («repeat» i t) t


/-- `prod α β` is the pointwise product of the components of `α` and `β` -/
def prod : ∀ {n}, TypeVec.{u} n → TypeVec.{u} n → TypeVec n
  | 0,     _, _ => Fin2.elim0
  | n + 1, α, β => (@prod n (drop α) (drop β)) ::: (last α × last β)


@[inherit_doc] scoped[MvFunctor] infixl:45 " ⊗ " => TypeVec.prod

/- porting note: the order of universes in `const` is reversed w.r.t. mathlib3 -/

/-- `const x α` is an arrow that ignores its source and constructs a `TypeVec` that
contains nothing but `x` -/
protected def const {β} (x : β) : ∀ {n} (α : TypeVec n), α ⟹ «repeat» _ β
  | succ _, α, Fin2.fs _ => TypeVec.const x (drop α) _
  | succ _, _, Fin2.fz   => fun _ => x


/-- vector of equality on a product of vectors -/
def repeatEq : ∀ {n} (α : TypeVec n), (α ⊗ α) ⟹ «repeat» _ Prop
  | 0, _ => nilFun
  | succ _, α => repeatEq (drop α) ::: uncurry Eq


theorem const_append1 {β γ} (x : γ) {n} (α : TypeVec n) :
    TypeVec.const x (α ::: β) = appendFun (TypeVec.const x α) fun _ => x := by
  /-
    β : Type u_1
    γ : Type u_2
    x : γ
    n : Nat
    α : TypeVec.{u_1} n
    ⊢ Eq (TypeVec.const x (α.append1 β)) (TypeVec.appendFun (TypeVec.const x α) fu …
  -/
                         /-
                           🎉 no goals
                         -/
  ext i : 1; cases i <;> rfl
                         /-
                           🎉 no goals
                         -/


theorem eq_nilFun {α β : TypeVec 0} (f : α ⟹ β) : f = nilFun := by
  /-
    α : TypeVec.{u_1} 0
    β : TypeVec.{u_2} 0
    f : α.Arrow β
    ⊢ Eq f TypeVec.nilFun
  -/
  ext x; cases x
         /-
           🎉 no goals
         -/


theorem id_eq_nilFun {α : TypeVec 0} : @id _ α = nilFun := by
  /-
    α : TypeVec.{u_1} 0
    ⊢ Eq TypeVec.id TypeVec.nilFun
  -/
  ext x; cases x
         /-
           🎉 no goals
         -/


theorem const_nil {β} (x : β) (α : TypeVec 0) : TypeVec.const x α = nilFun := by
  /-
    β : Type u_1
    x : β
    α : TypeVec.{u_2} 0
    ⊢ Eq (TypeVec.const x α) TypeVec.nilFun
  -/
  ext i : 1; cases i
             /-
               🎉 no goals
             -/


@[typevec]
theorem repeat_eq_append1 {β} {n} (α : TypeVec n) :
    repeatEq (α ::: β) = splitFun (α := (α ⊗ α) ::: _ )
    (α' := («repeat» n Prop) ::: _) (repeatEq α) (uncurry Eq) := by
  /-
    β : Type u_1
    n : Nat
    α : TypeVec.{u_1} n
    ⊢ Eq (α.append1 β).repeatEq (TypeVec.splitFun α.repeatEq (Function.uncurry Eq))
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> rfl
                  /-
                    🎉 no goals
                  -/


@[typevec]
                                                                  /-
                                                                    α : TypeVec.{u_1} 0
                                                                    ⊢ Eq α.repeatEq TypeVec.nilFun
                                                                  -/
theorem repeat_eq_nil (α : TypeVec 0) : repeatEq α = nilFun := by ext i; cases i
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


/-- predicate on a type vector to constrain only the last object -/
def PredLast' (α : TypeVec n) {β : Type*} (p : β → Prop) :
    (α ::: β) ⟹ «repeat» (n + 1) Prop :=
  splitFun (TypeVec.const True α) p


/-- predicate on the product of two type vectors to constrain only their last object -/
def RelLast' (α : TypeVec n) {β : Type*} (p : β → β → Prop) :
    (α ::: β) ⊗ (α ::: β) ⟹ «repeat» (n + 1) Prop :=
  splitFun (repeatEq α) (uncurry p)


/-- given `F : TypeVec.{u} (n+1) → Type u`, `curry F : Type u → TypeVec.{u} → Type u`,
i.e. its first argument can be fed in separately from the rest of the vector of arguments -/
def Curry (F : TypeVec.{u} (n + 1) → Type*) (α : Type u) (β : TypeVec.{u} n) : Type _ :=
  F (β ::: α)


instance Curry.inhabited (F : TypeVec.{u} (n + 1) → Type*) (α : Type u) (β : TypeVec.{u} n)
    [I : Inhabited (F <| (β ::: α))] : Inhabited (Curry F α β) :=
  I


/-- arrow to remove one element of a `repeat` vector -/
def dropRepeat (α : Type*) : ∀ {n}, drop («repeat» (succ n) α) ⟹ «repeat» n α
  | succ _, Fin2.fs i => dropRepeat α i
  | succ _, Fin2.fz   => fun (a : α) => a


/-- projection for a repeat vector -/
def ofRepeat {α : Sort _} : ∀ {n i}, «repeat» n α i → α
  | _, Fin2.fz   => fun (a : α) => a
  | _, Fin2.fs i => @ofRepeat _ _ i


theorem const_iff_true {α : TypeVec n} {i x p} : ofRepeat (TypeVec.const p α i x) ↔ p := by
  induction i with
  | fz      => rfl
  | fs _ ih => erw [TypeVec.const, @ih (drop α) x]



/-- left projection of a `prod` vector -/
def prod.fst : ∀ {n} {α β : TypeVec.{u} n}, α ⊗ β ⟹ α
  | succ _, α, β, Fin2.fs i => @prod.fst _ (drop α) (drop β) i
  | succ _, _, _, Fin2.fz => Prod.fst


/-- right projection of a `prod` vector -/
def prod.snd : ∀ {n} {α β : TypeVec.{u} n}, α ⊗ β ⟹ β
  | succ _, α, β, Fin2.fs i => @prod.snd _ (drop α) (drop β) i
  | succ _, _, _, Fin2.fz => Prod.snd


/-- introduce a product where both components are the same -/
def prod.diag : ∀ {n} {α : TypeVec.{u} n}, α ⟹ α ⊗ α
  | succ _, α, Fin2.fs _, x => @prod.diag _ (drop α) _ x
  | succ _, _, Fin2.fz, x => (x, x)


/-- constructor for `prod` -/
def prod.mk : ∀ {n} {α β : TypeVec.{u} n} (i : Fin2 n), α i → β i → (α ⊗ β) i
  | succ _, α, β, Fin2.fs i => mk (α := fun i => α i.fs) (β := fun i => β i.fs) i
  | succ _, _, _, Fin2.fz   => Prod.mk


@[simp]
theorem prod_fst_mk {α β : TypeVec n} (i : Fin2 n) (a : α i) (b : β i) :
    TypeVec.prod.fst i (prod.mk i a b) = a := by
  induction i with
  | fz => simp_all only [prod.fst, prod.mk]
  | fs _ i_ih => apply i_ih


@[simp]
theorem prod_snd_mk {α β : TypeVec n} (i : Fin2 n) (a : α i) (b : β i) :
    TypeVec.prod.snd i (prod.mk i a b) = b := by
  induction i with
  | fz => simp_all [prod.snd, prod.mk]
  | fs _ i_ih => apply i_ih


/-- `prod` is functorial -/
protected def prod.map : ∀ {n} {α α' β β' : TypeVec.{u} n}, α ⟹ β → α' ⟹ β' → α ⊗ α' ⟹ β ⊗ β'
  | succ _, α, α', β, β', x, y, Fin2.fs _, a =>
    @prod.map _ (drop α) (drop α') (drop β) (drop β') (dropFun x) (dropFun y) _ a
  | succ _, _, _, _, _, x, y, Fin2.fz, a => (x _ a.1, y _ a.2)




@[inherit_doc] scoped[MvFunctor] infixl:45 " ⊗' " => TypeVec.prod.map


theorem fst_prod_mk {α α' β β' : TypeVec n} (f : α ⟹ β) (g : α' ⟹ β') :
    TypeVec.prod.fst ⊚ (f ⊗' g) = f ⊚ TypeVec.prod.fst := by
  /-
    n : Nat
    α α' β β' : TypeVec.{u_1} n
    f : α.Arrow β
    g : α'.Arrow β'
    ⊢ Eq (TypeVec.comp TypeVec.prod.fst (TypeVec.prod.map f g)) (TypeVec.comp f Ty …
  -/
  funext i; induction i with
  | fz => rfl
  | fs _ i_ih => apply i_ih


theorem snd_prod_mk {α α' β β' : TypeVec n} (f : α ⟹ β) (g : α' ⟹ β') :
    TypeVec.prod.snd ⊚ (f ⊗' g) = g ⊚ TypeVec.prod.snd := by
  /-
    n : Nat
    α α' β β' : TypeVec.{u_1} n
    f : α.Arrow β
    g : α'.Arrow β'
    ⊢ Eq (TypeVec.comp TypeVec.prod.snd (TypeVec.prod.map f g)) (TypeVec.comp g Ty …
  -/
  funext i; induction i with
  | fz => rfl
  | fs _ i_ih => apply i_ih


theorem fst_diag {α : TypeVec n} : TypeVec.prod.fst ⊚ (prod.diag : α ⟹ _) = id := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    ⊢ Eq (TypeVec.comp TypeVec.prod.fst TypeVec.prod.diag) TypeVec.id
  -/
  funext i; induction i with
  | fz => rfl
  | fs _ i_ih => apply i_ih


theorem snd_diag {α : TypeVec n} : TypeVec.prod.snd ⊚ (prod.diag : α ⟹ _) = id := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    ⊢ Eq (TypeVec.comp TypeVec.prod.snd TypeVec.prod.diag) TypeVec.id
  -/
  funext i; induction i with
  | fz => rfl
  | fs _ i_ih => apply i_ih


theorem repeatEq_iff_eq {α : TypeVec n} {i x y} :
    ofRepeat (repeatEq α i (prod.mk _ x y)) ↔ x = y := by
  induction i with
  | fz => rfl
  | fs _ i_ih => erw [repeatEq, i_ih]


/-- given a predicate vector `p` over vector `α`, `Subtype_ p` is the type of vectors
that contain an `α` that satisfies `p` -/
def Subtype_ : ∀ {n} {α : TypeVec.{u} n}, (α ⟹ «repeat» n Prop) → TypeVec n
  | _, _, p, Fin2.fz => Subtype fun x => p Fin2.fz x
  | _, _, p, Fin2.fs i => Subtype_ (dropFun p) i


/-- projection on `Subtype_` -/
def subtypeVal : ∀ {n} {α : TypeVec.{u} n} (p : α ⟹ «repeat» n Prop), Subtype_ p ⟹ α
  | succ n, _, _, Fin2.fs i => @subtypeVal n _ _ i
  | succ _, _, _, Fin2.fz => Subtype.val


/-- arrow that rearranges the type of `Subtype_` to turn a subtype of vector into
a vector of subtypes -/
def toSubtype :
    ∀ {n} {α : TypeVec.{u} n} (p : α ⟹ «repeat» n Prop),
      (fun i : Fin2 n => { x // ofRepeat <| p i x }) ⟹ Subtype_ p
  | succ _, _, p, Fin2.fs i, x => toSubtype (dropFun p) i x
  | succ _, _, _, Fin2.fz, x => x


/-- arrow that rearranges the type of `Subtype_` to turn a vector of subtypes
into a subtype of vector -/
def ofSubtype {n} {α : TypeVec.{u} n} (p : α ⟹ «repeat» n Prop) :
    Subtype_ p ⟹ fun i : Fin2 n => { x // ofRepeat <| p i x }
  | Fin2.fs i, x => ofSubtype _ i x
  | Fin2.fz,   x => x


/-- similar to `toSubtype` adapted to relations (i.e. predicate on product) -/
def toSubtype' {n} {α : TypeVec.{u} n} (p : α ⊗ α ⟹ «repeat» n Prop) :
    (fun i : Fin2 n => { x : α i × α i // ofRepeat <| p i (prod.mk _ x.1 x.2) }) ⟹ Subtype_ p
  | Fin2.fs i, x => toSubtype' (dropFun p) i x
                                   /-
                                     n✝¹ n : Nat
                                     α✝ : TypeVec.{u} n
                                     p✝ : (α✝.prod α✝).Arrow (TypeVec.repeat n Prop)
                                     n✝ : Nat
                                     α : TypeVec.{u} (HAdd.hAdd n✝ 1)
                                     p : (α.prod α).Arrow (TypeVec.repeat (HAdd.hAdd n✝ 1) Prop)
                                     x : (fun i => Subtype fun x => TypeVec.ofRepeat (p i (TypeVec.prod.mk i x.1 x. …
                                     ⊢ Eq (TypeVec.ofRepeat (p Fin2.fz (TypeVec.prod.mk Fin2.fz (↑x).1 (↑x).2))) (p …
                                   -/
  | Fin2.fz, x => ⟨x.val, cast (by congr) x.property⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- similar to `of_subtype` adapted to relations (i.e. predicate on product) -/
def ofSubtype' {n} {α : TypeVec.{u} n} (p : α ⊗ α ⟹ «repeat» n Prop) :
    Subtype_ p ⟹ fun i : Fin2 n => { x : α i × α i // ofRepeat <| p i (prod.mk _ x.1 x.2) }
  | Fin2.fs i, x => ofSubtype' _ i x
                                   /-
                                     n✝¹ n : Nat
                                     α✝ : TypeVec.{u} n
                                     p✝ : (α✝.prod α✝).Arrow (TypeVec.repeat n Prop)
                                     n✝ : Nat
                                     α : TypeVec.{u} (HAdd.hAdd n✝ 1)
                                     p : (α.prod α).Arrow (TypeVec.repeat (HAdd.hAdd n✝ 1) Prop)
                                     x : TypeVec.Subtype_ p Fin2.fz
                                     ⊢ Eq (p Fin2.fz ↑x) (TypeVec.ofRepeat (p Fin2.fz (TypeVec.prod.mk Fin2.fz (↑x) …
                                   -/
  | Fin2.fz, x => ⟨x.val, cast (by congr) x.property⟩
                                   /-
                                     🎉 no goals
                                   -/


/-- similar to `diag` but the target vector is a `Subtype_`
guaranteeing the equality of the components -/
def diagSub {n} {α : TypeVec.{u} n} : α ⟹ Subtype_ (repeatEq α)
  | Fin2.fs _, x => @diagSub _ (drop α) _ x
  | Fin2.fz, x => ⟨(x, x), rfl⟩


theorem subtypeVal_nil {α : TypeVec.{u} 0} (ps : α ⟹ «repeat» 0 Prop) :
    TypeVec.subtypeVal ps = nilFun :=
               /-
                 α : TypeVec.{u} 0
                 ps : α.Arrow (TypeVec.repeat 0 Prop)
                 ⊢ ∀ (x : Fin2 0), Eq (TypeVec.subtypeVal ps x) (TypeVec.nilFun x)
               -/
  funext <| by rintro ⟨⟩
               /-
                 🎉 no goals
               -/


theorem diag_sub_val {n} {α : TypeVec.{u} n} : subtypeVal (repeatEq α) ⊚ diagSub = prod.diag := by
  /-
    n : Nat
    α : TypeVec.{u} n
    ⊢ Eq (TypeVec.comp (TypeVec.subtypeVal α.repeatEq) TypeVec.diagSub) TypeVec.pr …
  -/
  ext i x
  induction i with
  | fz => simp only [comp, subtypeVal, repeatEq.eq_2, diagSub, prod.diag]
  | fs _ i_ih => apply @i_ih (drop α)


theorem prod_id : ∀ {n} {α β : TypeVec.{u} n}, (id ⊗' id) = (id : α ⊗ β ⟹ _) := by
  /-
    ⊢ ∀ {n : Nat} {α β : TypeVec.{u} n}, Eq (TypeVec.prod.map TypeVec.id TypeVec.i …
  -/
  intros
  /-
    n✝ : Nat
    α✝ β✝ : TypeVec.{u} n✝
    ⊢ Eq (TypeVec.prod.map TypeVec.id TypeVec.id) TypeVec.id
  -/
  ext i a
  induction i with
  | fz => cases a; rfl
  | fs _ i_ih => apply i_ih


theorem append_prod_appendFun {n} {α α' β β' : TypeVec.{u} n} {φ φ' ψ ψ' : Type u}
    {f₀ : α ⟹ α'} {g₀ : β ⟹ β'} {f₁ : φ → φ'} {g₁ : ψ → ψ'} :
    ((f₀ ⊗' g₀) ::: (_root_.Prod.map f₁ g₁)) = ((f₀ ::: f₁) ⊗' (g₀ ::: g₁)) := by
  /-
    n : Nat
    α α' β β' : TypeVec.{u} n
    φ φ' ψ ψ' : Type u
    f₀ : α.Arrow α'
    g₀ : β.Arrow β'
    f₁ : φ → φ'
    g₁ : ψ → ψ'
    ⊢ Eq (TypeVec.appendFun (TypeVec.prod.map f₀ g₀) (Prod.map f₁ g₁)) (TypeVec.pr …
  -/
  ext i a
  /-
    case a.h
    n : Nat
    α α' β β' : TypeVec.{u} n
    φ φ' ψ ψ' : Type u
    f₀ : α.Arrow α'
    g₀ : β.Arrow β'
    f₁ : φ → φ'
    g₁ : ψ → ψ'
    i : Fin2 (HAdd.hAdd n 1)
    a : (α.prod β).append1 (Prod φ ψ) i
    ⊢ Eq (TypeVec.appendFun (TypeVec.prod.map f₀ g₀) (Prod.map f₁ g₁) i a) (TypeVe …
  -/
  cases i
    /-
      case a.h.fz
      n : Nat
      α α' β β' : TypeVec.{u} n
      φ φ' ψ ψ' : Type u
      f₀ : α.Arrow α'
      g₀ : β.Arrow β'
      f₁ : φ → φ'
      g₁ : ψ → ψ'
      a : (α.prod β).append1 (Prod φ ψ) Fin2.fz
      ⊢ Eq (TypeVec.appendFun (TypeVec.prod.map f₀ g₀) (Prod.map f₁ g₁) Fin2.fz a) ( …
    -/
  · cases a
    /-
      case a.h.fz.mk
      n : Nat
      α α' β β' : TypeVec.{u} n
      φ φ' ψ ψ' : Type u
      f₀ : α.Arrow α'
      g₀ : β.Arrow β'
      f₁ : φ → φ'
      g₁ : ψ → ψ'
      fst✝ : φ
      snd✝ : ψ
      ⊢ Eq (TypeVec.appendFun (TypeVec.prod.map f₀ g₀) (Prod.map f₁ g₁) Fin2.fz { fs …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case a.h.fs
      n : Nat
      α α' β β' : TypeVec.{u} n
      φ φ' ψ ψ' : Type u
      f₀ : α.Arrow α'
      g₀ : β.Arrow β'
      f₁ : φ → φ'
      g₁ : ψ → ψ'
      a✝ : Fin2 n
      a : (α.prod β).append1 (Prod φ ψ) a✝.fs
      ⊢ Eq (TypeVec.appendFun (TypeVec.prod.map f₀ g₀) (Prod.map f₁ g₁) a✝.fs a) (Ty …
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem dropFun_diag {α} : dropFun (@prod.diag (n + 1) α) = prod.diag := by
  /-
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    ⊢ Eq (TypeVec.dropFun TypeVec.prod.diag) TypeVec.prod.diag
  -/
  ext i : 2
  /-
    case a.h
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    i : Fin2 n
    x✝ : α.drop i
    ⊢ Eq (TypeVec.dropFun TypeVec.prod.diag i x✝) (TypeVec.prod.diag i x✝)
  -/
                                        /-
                                          🎉 no goals
                                        -/
  induction i <;> simp [dropFun, *] <;> rfl
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem dropFun_subtypeVal {α} (p : α ⟹ «repeat» (n + 1) Prop) :
    dropFun (subtypeVal p) = subtypeVal _ :=
  rfl


@[simp]
theorem lastFun_subtypeVal {α} (p : α ⟹ «repeat» (n + 1) Prop) :
    lastFun (subtypeVal p) = Subtype.val :=
  rfl


@[simp]
theorem dropFun_toSubtype {α} (p : α ⟹ «repeat» (n + 1) Prop) :
    dropFun (toSubtype p) = toSubtype _ := by
  /-
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n 1) Prop)
    ⊢ Eq (TypeVec.dropFun (TypeVec.toSubtype p)) (TypeVec.toSubtype fun i => p i.fs)
  -/
  ext i
  /-
    case a.h
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n 1) Prop)
    i : Fin2 n
    x✝ : TypeVec.drop (fun i => Subtype fun x => TypeVec.ofRepeat (p i x)) i
    ⊢ Eq (TypeVec.dropFun (TypeVec.toSubtype p) i x✝) (TypeVec.toSubtype (fun i => …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  induction i <;> simp [dropFun, *] <;> rfl
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem lastFun_toSubtype {α} (p : α ⟹ «repeat» (n + 1) Prop) :
    lastFun (toSubtype p) = _root_.id := by
  /-
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n 1) Prop)
    ⊢ Eq (TypeVec.lastFun (TypeVec.toSubtype p)) _root_.id
  -/
  ext i : 2
  /-
    case h
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n 1) Prop)
    i : TypeVec.last fun i => Subtype fun x => TypeVec.ofRepeat (p i x)
    ⊢ Eq (TypeVec.lastFun (TypeVec.toSubtype p) i) (_root_.id i)
  -/
  induction i; simp [dropFun, *]; rfl
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem dropFun_of_subtype {α} (p : α ⟹ «repeat» (n + 1) Prop) :
    dropFun (ofSubtype p) = ofSubtype _ := by
  /-
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n 1) Prop)
    ⊢ Eq (TypeVec.dropFun (TypeVec.ofSubtype p)) (TypeVec.ofSubtype (TypeVec.dropF …
  -/
  ext i : 2
  /-
    case a.h
    n : Nat
    α : TypeVec.{u_1} (HAdd.hAdd n 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n 1) Prop)
    i : Fin2 n
    x✝ : (TypeVec.Subtype_ p).drop i
    ⊢ Eq (TypeVec.dropFun (TypeVec.ofSubtype p) i x✝) (TypeVec.ofSubtype (TypeVec. …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  induction i <;> simp [dropFun, *] <;> rfl
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem lastFun_of_subtype {α} (p : α ⟹ «repeat» (n + 1) Prop) :
    lastFun (ofSubtype p) = _root_.id := rfl


@[simp]
theorem dropFun_RelLast' {α : TypeVec n} {β} (R : β → β → Prop) :
    dropFun (RelLast' α R) = repeatEq α :=
  rfl


@[simp]
theorem dropFun_prod {α α' β β' : TypeVec (n + 1)} (f : α ⟹ β) (f' : α' ⟹ β') :
    dropFun (f ⊗' f') = (dropFun f ⊗' dropFun f') := by
  /-
    n : Nat
    α α' β β' : TypeVec.{u_1} (HAdd.hAdd n 1)
    f : α.Arrow β
    f' : α'.Arrow β'
    ⊢ Eq (TypeVec.dropFun (TypeVec.prod.map f f')) (TypeVec.prod.map (TypeVec.drop …
  -/
  ext i : 2
  /-
    case a.h
    n : Nat
    α α' β β' : TypeVec.{u_1} (HAdd.hAdd n 1)
    f : α.Arrow β
    f' : α'.Arrow β'
    i : Fin2 n
    x✝ : (α.prod α').drop i
    ⊢ Eq (TypeVec.dropFun (TypeVec.prod.map f f') i x✝) (TypeVec.prod.map (TypeVec …
  -/
                                        /-
                                          🎉 no goals
                                        -/
  induction i <;> simp [dropFun, *] <;> rfl
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem lastFun_prod {α α' β β' : TypeVec (n + 1)} (f : α ⟹ β) (f' : α' ⟹ β') :
    lastFun (f ⊗' f') = Prod.map (lastFun f) (lastFun f') := by
  /-
    n : Nat
    α α' β β' : TypeVec.{u_1} (HAdd.hAdd n 1)
    f : α.Arrow β
    f' : α'.Arrow β'
    ⊢ Eq (TypeVec.lastFun (TypeVec.prod.map f f')) (Prod.map (TypeVec.lastFun f) ( …
  -/
  ext i : 1
  /-
    case h
    n : Nat
    α α' β β' : TypeVec.{u_1} (HAdd.hAdd n 1)
    f : α.Arrow β
    f' : α'.Arrow β'
    i : (α.prod α').last
    ⊢ Eq (TypeVec.lastFun (TypeVec.prod.map f f') i) (Prod.map (TypeVec.lastFun f) …
  -/
  induction i; simp [lastFun, *]; rfl
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem dropFun_from_append1_drop_last {α : TypeVec (n + 1)} :
    dropFun (@fromAppend1DropLast _ α) = id :=
  rfl


@[simp]
theorem lastFun_from_append1_drop_last {α : TypeVec (n + 1)} :
    lastFun (@fromAppend1DropLast _ α) = _root_.id :=
  rfl


@[simp]
theorem dropFun_id {α : TypeVec (n + 1)} : dropFun (@TypeVec.id _ α) = id :=
  rfl


@[simp]
theorem prod_map_id {α β : TypeVec n} : (@TypeVec.id _ α ⊗' @TypeVec.id _ β) = id := by
  /-
    n : Nat
    α β : TypeVec.{u_1} n
    ⊢ Eq (TypeVec.prod.map TypeVec.id TypeVec.id) TypeVec.id
  -/
  ext i x : 2
  /-
    case a.h
    n : Nat
    α β : TypeVec.{u_1} n
    i : Fin2 n
    x : α.prod β i
    ⊢ Eq (TypeVec.prod.map TypeVec.id TypeVec.id i x) (TypeVec.id i x)
  -/
  induction i <;> simp only [TypeVec.prod.map, *, dropFun_id]
  /-
    case a.h.fz
    n n✝ : Nat
    α β : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
    x : α.prod β Fin2.fz
    ⊢ Eq { fst := TypeVec.id Fin2.fz x.1, snd := TypeVec.id Fin2.fz x.2 } (TypeVec …
  -/
  cases x
    /-
      case a.h.fz.mk
      n n✝ : Nat
      α β : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
      fst✝ : α.last
      snd✝ : β.last
      ⊢ Eq { fst := TypeVec.id Fin2.fz { fst := fst✝, snd := snd✝ }.1, snd := TypeVe …
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case a.h.fs
      n n✝ : Nat
      a✝ : Fin2 n✝
      a_ih✝ : ∀ {α β : TypeVec.{u_1} n✝} (x : α.prod β a✝), Eq (TypeVec.prod.map Typ …
      α β : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
      x : α.prod β a✝.fs
      ⊢ Eq (TypeVec.id a✝ x) (TypeVec.id a✝.fs x)
    -/
  · rfl
    /-
      🎉 no goals
    -/


@[simp]
theorem subtypeVal_diagSub {α : TypeVec n} : subtypeVal (repeatEq α) ⊚ diagSub = prod.diag := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    ⊢ Eq (TypeVec.comp (TypeVec.subtypeVal α.repeatEq) TypeVec.diagSub) TypeVec.pr …
  -/
  ext i x
  induction i with
  | fz => simp [comp, diagSub, subtypeVal, prod.diag]
  | fs _ i_ih =>
    simp only [comp, subtypeVal, diagSub, prod.diag] at *
    apply i_ih


@[simp]
theorem toSubtype_of_subtype {α : TypeVec n} (p : α ⟹ «repeat» n Prop) :
    toSubtype p ⊚ ofSubtype p = id := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    p : α.Arrow (TypeVec.repeat n Prop)
    ⊢ Eq (TypeVec.comp (TypeVec.toSubtype p) (TypeVec.ofSubtype p)) TypeVec.id
  -/
  ext i x
  /-
    case a.h
    n : Nat
    α : TypeVec.{u_1} n
    p : α.Arrow (TypeVec.repeat n Prop)
    i : Fin2 n
    x : TypeVec.Subtype_ p i
    ⊢ Eq (TypeVec.comp (TypeVec.toSubtype p) (TypeVec.ofSubtype p) i x) (TypeVec.i …
  -/
                  /-
                    🎉 no goals
                  -/
  induction i <;> simp only [id, toSubtype, comp, ofSubtype] at *
  /-
    case a.h.fs
    n n✝ : Nat
    a✝ : Fin2 n✝
    a_ih✝ : ∀ {α : TypeVec.{u_1} n✝} (p : α.Arrow (TypeVec.repeat n✝ Prop)) (x : T …
    α : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n✝ 1) Prop)
    x : TypeVec.Subtype_ p a✝.fs
    ⊢ Eq (TypeVec.toSubtype (TypeVec.dropFun p) a✝ (TypeVec.ofSubtype (TypeVec.dro …
  -/
  simp [*]
  /-
    🎉 no goals
  -/


@[simp]
theorem subtypeVal_toSubtype {α : TypeVec n} (p : α ⟹ «repeat» n Prop) :
    subtypeVal p ⊚ toSubtype p = fun _ => Subtype.val := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    p : α.Arrow (TypeVec.repeat n Prop)
    ⊢ Eq (TypeVec.comp (TypeVec.subtypeVal p) (TypeVec.toSubtype p)) fun x => Subt …
  -/
  ext i x
  /-
    case a.h
    n : Nat
    α : TypeVec.{u_1} n
    p : α.Arrow (TypeVec.repeat n Prop)
    i : Fin2 n
    x : (fun i => Subtype fun x => TypeVec.ofRepeat (p i x)) i
    ⊢ Eq (TypeVec.comp (TypeVec.subtypeVal p) (TypeVec.toSubtype p) i x) ↑x
  -/
                  /-
                    🎉 no goals
                  -/
  induction i <;> simp only [toSubtype, comp, subtypeVal] at *
  /-
    case a.h.fs
    n n✝ : Nat
    a✝ : Fin2 n✝
    a_ih✝ : ∀ {α : TypeVec.{u_1} n✝} (p : α.Arrow (TypeVec.repeat n✝ Prop)) (x : S …
    α : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
    p : α.Arrow (TypeVec.repeat (HAdd.hAdd n✝ 1) Prop)
    x : Subtype fun x => TypeVec.ofRepeat (p a✝.fs x)
    ⊢ Eq (TypeVec.subtypeVal (TypeVec.dropFun p) a✝ (TypeVec.toSubtype (TypeVec.dr …
  -/
  simp [*]
  /-
    🎉 no goals
  -/


@[simp]
theorem toSubtype_of_subtype_assoc
    {α β : TypeVec n} (p : α ⟹ «repeat» n Prop) (f : β ⟹ Subtype_ p) :
    @toSubtype n _ p ⊚ ofSubtype _ ⊚ f = f := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    β : TypeVec.{u_2} n
    p : α.Arrow (TypeVec.repeat n Prop)
    f : β.Arrow (TypeVec.Subtype_ p)
    ⊢ Eq (TypeVec.comp (TypeVec.toSubtype p) (TypeVec.comp (TypeVec.ofSubtype p) f …
  -/
  rw [← comp_assoc, toSubtype_of_subtype]; simp
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
theorem toSubtype'_of_subtype' {α : TypeVec n} (r : α ⊗ α ⟹ «repeat» n Prop) :
    toSubtype' r ⊚ ofSubtype' r = id := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    r : (α.prod α).Arrow (TypeVec.repeat n Prop)
    ⊢ Eq (TypeVec.comp (TypeVec.toSubtype' r) (TypeVec.ofSubtype' r)) TypeVec.id
  -/
  ext i x
  /-
    case a.h
    n : Nat
    α : TypeVec.{u_1} n
    r : (α.prod α).Arrow (TypeVec.repeat n Prop)
    i : Fin2 n
    x : TypeVec.Subtype_ r i
    ⊢ Eq (TypeVec.comp (TypeVec.toSubtype' r) (TypeVec.ofSubtype' r) i x) (TypeVec …
  -/
  induction i
      /-
        case a.h.fz
        n n✝ : Nat
        α : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
        r : (α.prod α).Arrow (TypeVec.repeat (HAdd.hAdd n✝ 1) Prop)
        x : TypeVec.Subtype_ r Fin2.fz
        ⊢ Eq (TypeVec.comp (TypeVec.toSubtype' r) (TypeVec.ofSubtype' r) Fin2.fz x) (T …
      -/
  <;> dsimp only [id, toSubtype', comp, ofSubtype'] at *
      /-
        case a.h.fz
        n n✝ : Nat
        α : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
        r : (α.prod α).Arrow (TypeVec.repeat (HAdd.hAdd n✝ 1) Prop)
        x : TypeVec.Subtype_ r Fin2.fz
        ⊢ Eq ⟨↑x, ⋯⟩ x
      -/
      /-
        🎉 no goals
      -/
  <;> simp [Subtype.eta, *]
      /-
        🎉 no goals
      -/


theorem subtypeVal_toSubtype' {α : TypeVec n} (r : α ⊗ α ⟹ «repeat» n Prop) :
    subtypeVal r ⊚ toSubtype' r = fun i x => prod.mk i x.1.fst x.1.snd := by
  /-
    n : Nat
    α : TypeVec.{u_1} n
    r : (α.prod α).Arrow (TypeVec.repeat n Prop)
    ⊢ Eq (TypeVec.comp (TypeVec.subtypeVal r) (TypeVec.toSubtype' r)) fun i x => T …
  -/
  ext i x
  /-
    case a.h
    n : Nat
    α : TypeVec.{u_1} n
    r : (α.prod α).Arrow (TypeVec.repeat n Prop)
    i : Fin2 n
    x : (fun i => Subtype fun x => TypeVec.ofRepeat (r i (TypeVec.prod.mk i x.1 x. …
    ⊢ Eq (TypeVec.comp (TypeVec.subtypeVal r) (TypeVec.toSubtype' r) i x) (TypeVec …
  -/
                  /-
                    🎉 no goals
                  -/
  induction i <;> simp only [id, toSubtype', comp, subtypeVal, prod.mk] at *
  /-
    case a.h.fs
    n n✝ : Nat
    a✝ : Fin2 n✝
    a_ih✝ : ∀ {α : TypeVec.{u_1} n✝} (r : (α.prod α).Arrow (TypeVec.repeat n✝ Prop …
    α : TypeVec.{u_1} (HAdd.hAdd n✝ 1)
    r : (α.prod α).Arrow (TypeVec.repeat (HAdd.hAdd n✝ 1) Prop)
    x : Subtype fun x => TypeVec.ofRepeat (r a✝.fs (TypeVec.prod.mk a✝.fs x.1 x.2))
    ⊢ Eq (TypeVec.subtypeVal (TypeVec.dropFun r) a✝ (TypeVec.toSubtype' (TypeVec.d …
  -/
  simp [*]
  /-
    🎉 no goals
  -/


