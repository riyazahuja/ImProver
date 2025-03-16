/-- Free monoid over a given alphabet. -/
@[to_additive "Free nonabelian additive monoid over a given alphabet"]
def FreeMonoid (α) := List α


/-- The identity equivalence between `FreeMonoid α` and `List α`. -/
@[to_additive "The identity equivalence between `FreeAddMonoid α` and `List α`."]
def toList : FreeMonoid α ≃ List α := Equiv.refl _


/-- The identity equivalence between `List α` and `FreeMonoid α`. -/
@[to_additive "The identity equivalence between `List α` and `FreeAddMonoid α`."]
def ofList : List α ≃ FreeMonoid α := Equiv.refl _


@[to_additive (attr := simp)]
theorem toList_symm : (@toList α).symm = ofList := rfl


@[to_additive (attr := simp)]
theorem ofList_symm : (@ofList α).symm = toList := rfl


@[to_additive (attr := simp)]
theorem toList_ofList (l : List α) : toList (ofList l) = l := rfl


@[to_additive (attr := simp)]
theorem ofList_toList (xs : FreeMonoid α) : ofList (toList xs) = xs := rfl


@[to_additive (attr := simp)]
theorem toList_comp_ofList : @toList α ∘ ofList = id := rfl


@[to_additive (attr := simp)]
theorem ofList_comp_toList : @ofList α ∘ toList = id := rfl


@[to_additive]
instance : CancelMonoid (FreeMonoid α) where
  one := ofList []
  mul x y := ofList (toList x ++ toList y)
  mul_one := List.append_nil
  one_mul := List.nil_append
  mul_assoc := List.append_assoc
  mul_left_cancel _ _ _ := List.append_cancel_left
  mul_right_cancel _ _ _ := List.append_cancel_right


@[to_additive]
instance : Inhabited (FreeMonoid α) := ⟨1⟩


@[to_additive]
instance [IsEmpty α] : Unique (FreeMonoid α) := inferInstanceAs <| Unique (List α)


@[to_additive (attr := simp)]
theorem toList_one : toList (1 : FreeMonoid α) = [] := rfl


@[to_additive (attr := simp)]
theorem ofList_nil : ofList ([] : List α) = 1 := rfl


@[to_additive (attr := simp)]
theorem toList_mul (xs ys : FreeMonoid α) : toList (xs * ys) = toList xs ++ toList ys := rfl


@[to_additive (attr := simp)]
theorem ofList_append (xs ys : List α) : ofList (xs ++ ys) = ofList xs * ofList ys := rfl


@[to_additive (attr := simp)]
theorem toList_prod (xs : List (FreeMonoid α)) : toList xs.prod = (xs.map toList).flatten := by
  /-
    α : Type u_1
    xs : List (FreeMonoid α)
    ⊢ Eq (FreeMonoid.toList xs.prod) (List.map (⇑FreeMonoid.toList) xs).flatten
  -/
                   /-
                     🎉 no goals
                   -/
  induction xs <;> simp [*, List.flatten]
                   /-
                     🎉 no goals
                   -/


@[to_additive (attr := simp)]
theorem ofList_flatten (xs : List (List α)) : ofList xs.flatten = (xs.map ofList).prod :=
                         /-
                           α : Type u_1
                           xs : List (List α)
                           ⊢ Eq (FreeMonoid.toList (FreeMonoid.ofList xs.flatten)) (FreeMonoid.toList (Li …
                         -/
  toList.injective <| by simp
                         /-
                           🎉 no goals
                         -/


@[deprecated (since := "2024-10-15")] alias ofList_join := ofList_flatten

@[deprecated (since := "2024-10-15")]
alias _root_.FreeAddMonoid.ofList_join := _root_.FreeAddMonoid.ofList_flatten


/-- Embeds an element of `α` into `FreeMonoid α` as a singleton list. -/
@[to_additive "Embeds an element of `α` into `FreeAddMonoid α` as a singleton list."]
def of (x : α) : FreeMonoid α := ofList [x]


@[to_additive (attr := simp)]
theorem toList_of (x : α) : toList (of x) = [x] := rfl


@[to_additive]
theorem ofList_singleton (x : α) : ofList [x] = of x := rfl


@[to_additive (attr := simp)]
theorem ofList_cons (x : α) (xs : List α) : ofList (x :: xs) = of x * ofList xs := rfl


@[to_additive]
theorem toList_of_mul (x : α) (xs : FreeMonoid α) : toList (of x * xs) = x :: toList xs := rfl


@[to_additive]
theorem of_injective : Function.Injective (@of α) := List.singleton_injective


/-- The length of a free monoid element: 1.length = 0 and (a * b).length = a.length + b.length -/
@[to_additive "The length of an additive free monoid element: 1.length = 0 and (a + b).length =
  a.length + b.length"]
def length (a : FreeMonoid α) : ℕ := List.length a


@[to_additive (attr := simp)]
theorem length_one : length (1 : FreeMonoid α) = 0 := rfl


@[to_additive (attr := simp)]
theorem length_eq_zero : length a = 0 ↔ a = 1 := List.length_eq_zero


@[to_additive (attr := simp)]
theorem length_of (m : α) : length (of m) = 1 := rfl


@[to_additive existing]
theorem length_eq_one : length a = 1 ↔ ∃ m, a = FreeMonoid.of m :=
  List.length_eq_one


@[to_additive]
theorem length_eq_two {v : FreeMonoid α} :
    v.length = 2 ↔ ∃ c d, v = FreeMonoid.of c * FreeMonoid.of d := List.length_eq_two


@[to_additive]
theorem length_eq_three {v : FreeMonoid α} : v.length = 3 ↔ ∃ (a b c : α), v = of a * of b * of c :=
  List.length_eq_three


@[to_additive (attr := simp)]
theorem length_mul (a b : FreeMonoid α) : (a * b).length = a.length + b.length :=
  List.length_append _ _


@[to_additive (attr := simp)]
theorem of_ne_one (a : α) : of a ≠ 1 := by
  /-
    α : Type u_1
    a : α
    ⊢ Ne (FreeMonoid.of a) 1
  -/
  intro h
  /-
    α : Type u_1
    a : α
    h : Eq (FreeMonoid.of a) 1
    ⊢ False
  -/
  have := congrArg FreeMonoid.length h
  /-
    α : Type u_1
    a : α
    h : Eq (FreeMonoid.of a) 1
    this : Eq (FreeMonoid.of a).length (FreeMonoid.length 1)
    ⊢ False
  -/
  simp only [length_of, length_one, Nat.succ_ne_self] at this
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem one_ne_of (a : α) : 1 ≠ of a := of_ne_one _ |>.symm


/-- Membership in a free monoid element -/
@[to_additive "Membership in a free monoid element"]
def mem (a : FreeMonoid α) (m : α) := m ∈ toList a


@[to_additive]
instance : Membership α (FreeMonoid α) := ⟨mem⟩


@[to_additive]
theorem not_mem_one : ¬ m ∈ (1 : FreeMonoid α) := List.not_mem_nil _


@[to_additive (attr := simp)]
theorem mem_of {n : α} : m ∈ of n ↔ m = n := List.mem_singleton


@[to_additive]
theorem mem_of_self : m ∈ of m := List.mem_singleton_self _


@[to_additive (attr := simp)]
theorem mem_mul {a b : FreeMonoid α} : m ∈ (a * b) ↔ m ∈ a ∨ m ∈ b := List.mem_append


/-- Recursor for `FreeMonoid` using `1` and `FreeMonoid.of x * xs` instead of `[]` and `x :: xs`. -/
@[to_additive (attr := elab_as_elim, induction_eliminator)
  "Recursor for `FreeAddMonoid` using `0` and
  FreeAddMonoid.of x + xs` instead of `[]` and `x :: xs`."]
-- Porting note: change from `List.recOn` to `List.rec` since only the latter is computable
def recOn {C : FreeMonoid α → Sort*} (xs : FreeMonoid α) (h0 : C 1)
    (ih : ∀ x xs, C xs → C (of x * xs)) : C xs := List.rec h0 ih xs


@[to_additive (attr := simp)]
theorem recOn_one {C : FreeMonoid α → Sort*} (h0 : C 1) (ih : ∀ x xs, C xs → C (of x * xs)) :
    @recOn α C 1 h0 ih = h0 := rfl


@[to_additive (attr := simp)]
theorem recOn_of_mul {C : FreeMonoid α → Sort*} (x : α) (xs : FreeMonoid α) (h0 : C 1)
    (ih : ∀ x xs, C xs → C (of x * xs)) : @recOn α C (of x * xs) h0 ih = ih x xs (recOn xs h0 ih) :=
  rfl


/-- An induction principle on free monoids, with cases for `1`, `FreeMonoid.of` and `*`. -/
@[to_additive (attr := elab_as_elim, induction_eliminator)
"An induction principle on free monoids, with cases for `0`, `FreeAddMonoid.of` and `+`."]
protected theorem inductionOn {C : FreeMonoid α → Prop} (z : FreeMonoid α) (one : C 1)
    (of : ∀ (x : α), C (FreeMonoid.of x)) (mul : ∀ (x y : FreeMonoid α), C x → C y → C (x * y)) :
  C z := List.rec one (fun _ _ ih => mul [_] _ (of _) ih) z


/-- An induction principle for free monoids which mirrors induction on lists, with cases analogous
to the empty list and cons -/
@[to_additive (attr := elab_as_elim) "An induction principle for free monoids which mirrors
induction on lists, with cases analogous to the empty list and cons"]
protected theorem inductionOn' {p : FreeMonoid α → Prop} (a : FreeMonoid α)
    (one : p (1 : FreeMonoid α)) (mul_of : ∀ b a, p a → p (of b * a)) : p a :=
  List.rec one (fun _ _ tail_ih => mul_of _ _ tail_ih) a


/-- A version of `List.cases_on` for `FreeMonoid` using `1` and `FreeMonoid.of x * xs` instead of
`[]` and `x :: xs`. -/
@[to_additive (attr := elab_as_elim, cases_eliminator)
  "A version of `List.casesOn` for `FreeAddMonoid` using `0` and
  `FreeAddMonoid.of x + xs` instead of `[]` and `x :: xs`."]
def casesOn {C : FreeMonoid α → Sort*} (xs : FreeMonoid α) (h0 : C 1)
    (ih : ∀ x xs, C (of x * xs)) : C xs := List.casesOn xs h0 ih


@[to_additive (attr := simp)]
theorem casesOn_one {C : FreeMonoid α → Sort*} (h0 : C 1) (ih : ∀ x xs, C (of x * xs)) :
    @casesOn α C 1 h0 ih = h0 := rfl


@[to_additive (attr := simp)]
theorem casesOn_of_mul {C : FreeMonoid α → Sort*} (x : α) (xs : FreeMonoid α) (h0 : C 1)
    (ih : ∀ x xs, C (of x * xs)) : @casesOn α C (of x * xs) h0 ih = ih x xs := rfl


@[to_additive (attr := ext)]
theorem hom_eq ⦃f g : FreeMonoid α →* M⦄ (h : ∀ x, f (of x) = g (of x)) : f = g :=
  MonoidHom.ext fun l ↦ recOn l (f.map_one.trans g.map_one.symm)
                       /-
                         α : Type u_1
                         M : Type u_4
                         inst✝ : Monoid M
                         f g : MonoidHom (FreeMonoid α) M
                         h : ∀ (x : α), Eq (f (FreeMonoid.of x)) (g (FreeMonoid.of x))
                         l : FreeMonoid α
                         x : α
                         xs : FreeMonoid α
                         hxs : Eq (f xs) (g xs)
                         ⊢ Eq (f (HMul.hMul (FreeMonoid.of x) xs)) (g (HMul.hMul (FreeMonoid.of x) xs))
                       -/
    (fun x xs hxs ↦ by simp only [h, hxs, MonoidHom.map_mul])
                       /-
                         🎉 no goals
                       -/


/-- A variant of `List.prod` that has `[x].prod = x` true definitionally.
The purpose is to make `FreeMonoid.lift_eval_of` true by `rfl`. -/
@[to_additive "A variant of `List.sum` that has `[x].sum = x` true definitionally.
The purpose is to make `FreeAddMonoid.lift_eval_of` true by `rfl`."]
def prodAux {M} [Monoid M] : List M → M
  | [] => 1
  | (x :: xs) => List.foldl (· * ·) x xs


@[to_additive]
lemma prodAux_eq : ∀ l : List M, FreeMonoid.prodAux l = l.prod
  | [] => rfl
                    /-
                      M : Type u_4
                      inst✝ : Monoid M
                      head✝ : M
                      xs : List M
                      ⊢ Eq (FreeMonoid.prodAux (List.cons head✝ xs)) (List.cons head✝ xs).prod
                    -/
  | (_ :: xs) => by simp [prodAux, List.prod_eq_foldl]
                    /-
                      🎉 no goals
                    -/


/-- Equivalence between maps `α → M` and monoid homomorphisms `FreeMonoid α →* M`. -/
@[to_additive "Equivalence between maps `α → A` and additive monoid homomorphisms
`FreeAddMonoid α →+ A`."]
def lift : (α → M) ≃ (FreeMonoid α →* M) where
  toFun f :=
  { toFun := fun l ↦ prodAux ((toList l).map f)
    map_one' := rfl
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               M : Type u_4
                               inst✝¹ : Monoid M
                               N : Type u_5
                               inst✝ : Monoid N
                               f : α → M
                               x✝¹ x✝ : FreeMonoid α
                               ⊢ Eq ({ toFun := fun l => FreeMonoid.prodAux (List.map f (FreeMonoid.toList l) …
                             -/
    map_mul' := fun _ _ ↦ by simp only [prodAux_eq, toList_mul, List.map_append, List.prod_append] }
                             /-
                               🎉 no goals
                             -/
  invFun f x := f (of x)
  left_inv _ := rfl
  right_inv _ := hom_eq fun _ ↦ rfl


@[to_additive (attr := simp)]
theorem lift_ofList (f : α → M) (l : List α) : lift f (ofList l) = (l.map f).prod :=
  prodAux_eq _


@[to_additive (attr := simp)]
theorem lift_symm_apply (f : FreeMonoid α →* M) : lift.symm f = f ∘ of := rfl


@[to_additive]
theorem lift_apply (f : α → M) (l : FreeMonoid α) : lift f l = ((toList l).map f).prod :=
  prodAux_eq _


@[to_additive]
theorem lift_comp_of (f : α → M) : lift f ∘ of = f := rfl


@[to_additive (attr := simp)]
theorem lift_eval_of (f : α → M) (x : α) : lift f (of x) = f x := rfl


@[to_additive (attr := simp)]
theorem lift_restrict (f : FreeMonoid α →* M) : lift (f ∘ of) = f := lift.apply_symm_apply f


@[to_additive]
theorem comp_lift (g : M →* N) (f : α → M) : g.comp (lift f) = lift (g ∘ f) := by
  /-
    α : Type u_1
    M : Type u_4
    inst✝¹ : Monoid M
    N : Type u_5
    inst✝ : Monoid N
    g : MonoidHom M N
    f : α → M
    ⊢ Eq (g.comp (FreeMonoid.lift f)) (FreeMonoid.lift (Function.comp (⇑g) f))
  -/
  ext
  /-
    case h
    α : Type u_1
    M : Type u_4
    inst✝¹ : Monoid M
    N : Type u_5
    inst✝ : Monoid N
    g : MonoidHom M N
    f : α → M
    x✝ : α
    ⊢ Eq ((g.comp (FreeMonoid.lift f)) (FreeMonoid.of x✝)) ((FreeMonoid.lift (Func …
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem hom_map_lift (g : M →* N) (f : α → M) (x : FreeMonoid α) : g (lift f x) = lift (g ∘ f) x :=
  DFunLike.ext_iff.1 (comp_lift g f) x


/-- Define a multiplicative action of `FreeMonoid α` on `β`. -/
@[to_additive "Define an additive action of `FreeAddMonoid α` on `β`."]
def mkMulAction (f : α → β → β) : MulAction (FreeMonoid α) β where
  smul l b := l.toList.foldr f b
  one_smul _ := rfl
  mul_smul _ _ _ := List.foldr_append _ _ _ _


@[to_additive]
theorem smul_def (f : α → β → β) (l : FreeMonoid α) (b : β) :
    haveI := mkMulAction f
    l • b = l.toList.foldr f b := rfl


@[to_additive]
theorem ofList_smul (f : α → β → β) (l : List α) (b : β) :
    haveI := mkMulAction f
    ofList l • b = l.foldr f b := rfl


@[to_additive (attr := simp)]
theorem of_smul (f : α → β → β) (x : α) (y : β) :
    (haveI := mkMulAction f
    of x • y) = f x y := rfl


/-- The unique monoid homomorphism `FreeMonoid α →* FreeMonoid β` that sends
each `of x` to `of (f x)`. -/
@[to_additive "The unique additive monoid homomorphism `FreeAddMonoid α →+ FreeAddMonoid β`
that sends each `of x` to `of (f x)`."]
def map (f : α → β) : FreeMonoid α →* FreeMonoid β where
  toFun l := ofList <| l.toList.map f
  map_one' := rfl
  map_mul' _ _ := List.map_append _ _ _


@[to_additive (attr := simp)]
theorem map_of (f : α → β) (x : α) : map f (of x) = of (f x) := rfl


@[to_additive]
theorem mem_map {m : β} : m ∈ map f a ↔ ∃ n ∈ a, f n = m := List.mem_map


@[to_additive]
theorem map_map {α₁ : Type*} {g : α₁ → α} {x : FreeMonoid α₁} :
    map f (map g x) = map (f ∘ g) x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    α₁ : Type u_6
    g : α₁ → α
    x : FreeMonoid α₁
    ⊢ Eq ((FreeMonoid.map f) ((FreeMonoid.map g) x)) ((FreeMonoid.map (Function.co …
  -/
  unfold map
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    α₁ : Type u_6
    g : α₁ → α
    x : FreeMonoid α₁
    ⊢ Eq ({ toFun := fun l => FreeMonoid.ofList (List.map f (FreeMonoid.toList l)) …
  -/
  simp only [MonoidHom.coe_mk, OneHom.coe_mk, toList_ofList, List.map_map]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem toList_map (f : α → β) (xs : FreeMonoid α) : toList (map f xs) = xs.toList.map f := rfl


@[to_additive]
theorem ofList_map (f : α → β) (xs : List α) : ofList (xs.map f) = map f (ofList xs) := rfl


@[to_additive]
theorem lift_of_comp_eq_map (f : α → β) : (lift fun x ↦ of (f x)) = map f := hom_eq fun _ ↦ rfl


@[to_additive]
theorem map_comp (g : β → γ) (f : α → β) : map (g ∘ f) = (map g).comp (map f) := hom_eq fun _ ↦ rfl


@[to_additive (attr := simp)]
theorem map_id : map (@id α) = MonoidHom.id (FreeMonoid α) := hom_eq fun _ ↦ rfl


/-- The only invertible element of the free monoid is 1; this instance enables `units_eq_one`. -/
@[to_additive]
instance uniqueUnits : Unique (FreeMonoid α)ˣ where
  uniq u := Units.ext <| toList.injective <|
    have : toList u.val ++ toList u.inv = [] := DFunLike.congr_arg toList u.val_inv
    (List.append_eq_nil.mp this).1


@[to_additive (attr := simp)]
theorem map_surjective {f : α → β} : Function.Surjective (map f) ↔ Function.Surjective f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ Iff (Function.Surjective ⇑(FreeMonoid.map f)) (Function.Surjective f)
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      f : α → β
      ⊢ Function.Surjective ⇑(FreeMonoid.map f) → Function.Surjective f
    -/
  · intro fs d
    /-
      case mp
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      ⊢ Exists fun a => Eq (f a) d
    -/
    rcases fs (FreeMonoid.of d) with ⟨b, hb⟩
    /-
      case mp.intro
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      b : FreeMonoid α
      hb : Eq ((FreeMonoid.map f) b) (FreeMonoid.of d)
      ⊢ Exists fun a => Eq (f a) d
    -/
    induction' b using FreeMonoid.inductionOn' with head _ _
      /-
        case mp.intro.one
        α : Type u_1
        β : Type u_2
        f : α → β
        fs : Function.Surjective ⇑(FreeMonoid.map f)
        d : β
        hb : Eq ((FreeMonoid.map f) 1) (FreeMonoid.of d)
        ⊢ Exists fun a => Eq (f a) d
      -/
    · have H := congr_arg length hb
      /-
        case mp.intro.one
        α : Type u_1
        β : Type u_2
        f : α → β
        fs : Function.Surjective ⇑(FreeMonoid.map f)
        d : β
        hb : Eq ((FreeMonoid.map f) 1) (FreeMonoid.of d)
        H : Eq ((FreeMonoid.map f) 1).length (FreeMonoid.of d).length
        ⊢ Exists fun a => Eq (f a) d
      -/
      simp only [length_one, length_of, Nat.zero_ne_one, map_one] at H
      /-
        🎉 no goals
      -/
    /-
      case mp.intro.mul_of
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      head : α
      a✝¹ : FreeMonoid α
      a✝ : Eq ((FreeMonoid.map f) a✝¹) (FreeMonoid.of d) → Exists fun a => Eq (f a) d
      hb : Eq ((FreeMonoid.map f) (HMul.hMul (FreeMonoid.of head) a✝¹)) (FreeMonoid. …
      ⊢ Exists fun a => Eq (f a) d
    -/
    simp only [map_mul, map_of] at hb
    /-
      case mp.intro.mul_of
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      head : α
      a✝¹ : FreeMonoid α
      a✝ : Eq ((FreeMonoid.map f) a✝¹) (FreeMonoid.of d) → Exists fun a => Eq (f a) d
      hb : Eq (HMul.hMul (FreeMonoid.of (f head)) ((FreeMonoid.map f) a✝¹)) (FreeMon …
      ⊢ Exists fun a => Eq (f a) d
    -/
    use head
    /-
      case h
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      head : α
      a✝¹ : FreeMonoid α
      a✝ : Eq ((FreeMonoid.map f) a✝¹) (FreeMonoid.of d) → Exists fun a => Eq (f a) d
      hb : Eq (HMul.hMul (FreeMonoid.of (f head)) ((FreeMonoid.map f) a✝¹)) (FreeMon …
      ⊢ Eq (f head) d
    -/
    have H := congr_arg length hb
    /-
      case h
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      head : α
      a✝¹ : FreeMonoid α
      a✝ : Eq ((FreeMonoid.map f) a✝¹) (FreeMonoid.of d) → Exists fun a => Eq (f a) d
      hb : Eq (HMul.hMul (FreeMonoid.of (f head)) ((FreeMonoid.map f) a✝¹)) (FreeMon …
      H : Eq (HMul.hMul (FreeMonoid.of (f head)) ((FreeMonoid.map f) a✝¹)).length (F …
      ⊢ Eq (f head) d
    -/
    simp only [length_mul, length_of, add_right_eq_self, length_eq_zero] at H
    /-
      case h
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      head : α
      a✝¹ : FreeMonoid α
      a✝ : Eq ((FreeMonoid.map f) a✝¹) (FreeMonoid.of d) → Exists fun a => Eq (f a) d
      hb : Eq (HMul.hMul (FreeMonoid.of (f head)) ((FreeMonoid.map f) a✝¹)) (FreeMon …
      H : Eq ((FreeMonoid.map f) a✝¹) 1
      ⊢ Eq (f head) d
    -/
    rw [H, mul_one] at hb
    /-
      case h
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective ⇑(FreeMonoid.map f)
      d : β
      head : α
      a✝¹ : FreeMonoid α
      a✝ : Eq ((FreeMonoid.map f) a✝¹) (FreeMonoid.of d) → Exists fun a => Eq (f a) d
      hb : Eq (FreeMonoid.of (f head)) (FreeMonoid.of d)
      H : Eq ((FreeMonoid.map f) a✝¹) 1
      ⊢ Eq (f head) d
    -/
    exact FreeMonoid.of_injective hb
    /-
      🎉 no goals
    -/
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    f : α → β
    ⊢ Function.Surjective f → Function.Surjective ⇑(FreeMonoid.map f)
  -/
  intro fs d
  /-
    case mpr
    α : Type u_1
    β : Type u_2
    f : α → β
    fs : Function.Surjective f
    d : FreeMonoid β
    ⊢ Exists fun a => Eq ((FreeMonoid.map f) a) d
  -/
  induction' d using FreeMonoid.inductionOn' with head tail ih
    /-
      case mpr.one
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective f
      ⊢ Exists fun a => Eq ((FreeMonoid.map f) a) 1
    -/
  · use 1
    /-
      case h
      α : Type u_1
      β : Type u_2
      f : α → β
      fs : Function.Surjective f
      ⊢ Eq ((FreeMonoid.map f) 1) 1
    -/
    rfl
    /-
      🎉 no goals
    -/
  /-
    case mpr.mul_of
    α : Type u_1
    β : Type u_2
    f : α → β
    fs : Function.Surjective f
    head : β
    tail : FreeMonoid β
    ih : Exists fun a => Eq ((FreeMonoid.map f) a) tail
    ⊢ Exists fun a => Eq ((FreeMonoid.map f) a) (HMul.hMul (FreeMonoid.of head) ta …
  -/
  specialize fs head
  /-
    case mpr.mul_of
    α : Type u_1
    β : Type u_2
    f : α → β
    head : β
    tail : FreeMonoid β
    ih : Exists fun a => Eq ((FreeMonoid.map f) a) tail
    fs : Exists fun a => Eq (f a) head
    ⊢ Exists fun a => Eq ((FreeMonoid.map f) a) (HMul.hMul (FreeMonoid.of head) ta …
  -/
  rcases fs with ⟨a, rfl⟩
  /-
    case mpr.mul_of.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    tail : FreeMonoid β
    ih : Exists fun a => Eq ((FreeMonoid.map f) a) tail
    a : α
    ⊢ Exists fun a_1 => Eq ((FreeMonoid.map f) a_1) (HMul.hMul (FreeMonoid.of (f a …
  -/
  rcases ih with ⟨b, rfl⟩
  /-
    case mpr.mul_of.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    a : α
    b : FreeMonoid α
    ⊢ Exists fun a_1 => Eq ((FreeMonoid.map f) a_1) (HMul.hMul (FreeMonoid.of (f a …
  -/
  use FreeMonoid.of a * b
  /-
    case h
    α : Type u_1
    β : Type u_2
    f : α → β
    a : α
    b : FreeMonoid α
    ⊢ Eq ((FreeMonoid.map f) (HMul.hMul (FreeMonoid.of a) b)) (HMul.hMul (FreeMono …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- reverses the symbols in a free monoid element -/
@[to_additive "reverses the symbols in an additive free monoid element"]
def reverse : FreeMonoid α → FreeMonoid α := List.reverse


@[to_additive (attr := simp)]
theorem reverse_of (a : α) : reverse (of a) = of a := rfl


@[to_additive]
theorem reverse_mul {a b : FreeMonoid α} : reverse (a * b) = reverse b * reverse a :=
  List.reverse_append _ _


@[to_additive (attr := simp)]
theorem reverse_reverse {a : FreeMonoid α} : reverse (reverse a) = a := by
  /-
    α : Type u_1
    a : FreeMonoid α
    ⊢ Eq a.reverse.reverse a
  -/
  apply List.reverse_reverse
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem length_reverse {a : FreeMonoid α} : a.reverse.length = a.length :=
  List.length_reverse _


