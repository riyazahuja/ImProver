/-- The minimal congruence relation `c` on `FreeMonoid (M ⊕ N)`
such that `FreeMonoid.of ∘ Sum.inl` and `FreeMonoid.of ∘ Sum.inr` are monoid homomorphisms
to the quotient by `c`. -/
@[to_additive "The minimal additive congruence relation `c` on `FreeAddMonoid (M ⊕ N)`
such that `FreeAddMonoid.of ∘ Sum.inl` and `FreeAddMonoid.of ∘ Sum.inr`
are additive monoid homomorphisms to the quotient by `c`."]
def coprodCon (M N : Type*) [MulOneClass M] [MulOneClass N] : Con (FreeMonoid (M ⊕ N)) :=
  sInf {c |
    (∀ x y : M, c (of (Sum.inl (x * y))) (of (Sum.inl x) * of (Sum.inl y)))
    ∧ (∀ x y : N, c (of (Sum.inr (x * y))) (of (Sum.inr x) * of (Sum.inr y)))
    ∧ c (of <| Sum.inl 1) 1 ∧ c (of <| Sum.inr 1) 1}


/-- Coproduct of two monoids or groups. -/
@[to_additive "Coproduct of two additive monoids or groups."]
def Coprod (M N : Type*) [MulOneClass M] [MulOneClass N] := (coprodCon M N).Quotient


@[inherit_doc]
scoped infix:30 " ∗ " => Coprod


@[to_additive] protected instance : MulOneClass (M ∗ N) := Con.mulOneClass _


/-- The natural projection `FreeMonoid (M ⊕ N) →* M ∗ N`. -/
@[to_additive "The natural projection `FreeAddMonoid (M ⊕ N) →+ AddMonoid.Coprod M N`."]
def mk : FreeMonoid (M ⊕ N) →* M ∗ N := Con.mk' _


@[to_additive (attr := simp)]
theorem con_ker_mk : Con.ker mk = coprodCon M N := Con.mk'_ker _


@[to_additive]
theorem mk_surjective : Surjective (@mk M N _ _) := Quot.mk_surjective


@[to_additive (attr := simp)]
theorem mrange_mk : MonoidHom.mrange (@mk M N _ _) = ⊤ := Con.mrange_mk'


@[to_additive]
theorem mk_eq_mk {w₁ w₂ : FreeMonoid (M ⊕ N)} : mk w₁ = mk w₂ ↔ coprodCon M N w₁ w₂ := Con.eq _


/-- The natural embedding `M →* M ∗ N`. -/
@[to_additive "The natural embedding `M →+ AddMonoid.Coprod M N`."]
def inl : M →* M ∗ N where
  toFun := fun x => mk (of (.inl x))
  map_one' := mk_eq_mk.2 fun _c hc => hc.2.2.1
  map_mul' := fun x y => mk_eq_mk.2 fun _c hc => hc.1 x y


/-- The natural embedding `N →* M ∗ N`. -/
@[to_additive "The natural embedding `N →+ AddMonoid.Coprod M N`."]
def inr : N →* M ∗ N where
  toFun := fun x => mk (of (.inr x))
  map_one' := mk_eq_mk.2 fun _c hc => hc.2.2.2
  map_mul' := fun x y => mk_eq_mk.2 fun _c hc => hc.2.1 x y


@[to_additive (attr := simp)]
theorem mk_of_inl (x : M) : (mk (of (.inl x)) : M ∗ N) = inl x := rfl


@[to_additive (attr := simp)]
theorem mk_of_inr (x : N) : (mk (of (.inr x)) : M ∗ N) = inr x := rfl


@[to_additive (attr := elab_as_elim)]
theorem induction_on' {C : M ∗ N → Prop} (m : M ∗ N)
    (one : C 1)
    (inl_mul : ∀ m x, C x → C (inl m * x))
    (inr_mul : ∀ n x, C x → C (inr n * x)) : C m := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : MulOneClass M
    inst✝ : MulOneClass N
    C : Monoid.Coprod M N → Prop
    m : Monoid.Coprod M N
    one : C 1
    inl_mul : ∀ (m : M) (x : Monoid.Coprod M N), C x → C (HMul.hMul (Monoid.Coprod …
    inr_mul : ∀ (n : N) (x : Monoid.Coprod M N), C x → C (HMul.hMul (Monoid.Coprod …
    ⊢ C m
  -/
  rcases mk_surjective m with ⟨x, rfl⟩
  induction x using FreeMonoid.inductionOn' with
  | one => exact one
  | mul_of x xs ih =>
    cases x with
    | inl m => simpa using inl_mul m _ ih
    | inr n => simpa using inr_mul n _ ih


@[to_additive (attr := elab_as_elim)]
theorem induction_on {C : M ∗ N → Prop} (m : M ∗ N)
    (inl : ∀ m, C (inl m)) (inr : ∀ n, C (inr n)) (mul : ∀ x y, C x → C y → C (x * y)) : C m :=
                      /-
                        M : Type u_1
                        N : Type u_2
                        inst✝¹ : MulOneClass M
                        inst✝ : MulOneClass N
                        C : Monoid.Coprod M N → Prop
                        m : Monoid.Coprod M N
                        inl : ∀ (m : M), C (Monoid.Coprod.inl m)
                        inr : ∀ (n : N), C (Monoid.Coprod.inr n)
                        mul : ∀ (x y : Monoid.Coprod M N), C x → C y → C (HMul.hMul x y)
                        ⊢ C 1
                      -/
  induction_on' m (by simpa using inl 1) (fun _ _ ↦ mul _ _ (inl _)) fun _ _ ↦ mul _ _ (inr _)
                      /-
                        🎉 no goals
                      -/


/-- Lift a monoid homomorphism `FreeMonoid (M ⊕ N) →* P` satisfying additional properties to
`M ∗ N →* P`. In many cases, `Coprod.lift` is more convenient.

Compared to `Coprod.lift`,
this definition allows a user to provide a custom computational behavior.
Also, it only needs `MulOneclass` assumptions while `Coprod.lift` needs a `Monoid` structure.
-/
@[to_additive "Lift an additive monoid homomorphism `FreeAddMonoid (M ⊕ N) →+ P` satisfying
additional properties to `AddMonoid.Coprod M N →+ P`.

Compared to `AddMonoid.Coprod.lift`,
this definition allows a user to provide a custom computational behavior.
Also, it only needs `AddZeroclass` assumptions
while `AddMonoid.Coprod.lift` needs an `AddMonoid` structure. "]
def clift (f : FreeMonoid (M ⊕ N) →* P)
    (hM₁ : f (of (.inl 1)) = 1) (hN₁ : f (of (.inr 1)) = 1)
    (hM : ∀ x y, f (of (.inl (x * y))) = f (of (.inl x) * of (.inl y)))
    (hN : ∀ x y, f (of (.inr (x * y))) = f (of (.inr x) * of (.inr y))) :
    M ∗ N →* P :=
  Con.lift _ f <| sInf_le ⟨hM, hN, hM₁.trans (map_one f).symm, hN₁.trans (map_one f).symm⟩


@[to_additive (attr := simp)]
theorem clift_apply_inl (f : FreeMonoid (M ⊕ N) →* P) (hM₁ hN₁ hM hN) (x : M) :
    clift f hM₁ hN₁ hM hN (inl x) = f (of (.inl x)) :=
  rfl


@[to_additive (attr := simp)]
theorem clift_apply_inr (f : FreeMonoid (M ⊕ N) →* P) (hM₁ hN₁ hM hN) (x : N) :
    clift f hM₁ hN₁ hM hN (inr x) = f (of (.inr x)) :=
  rfl


@[to_additive (attr := simp)]
theorem clift_apply_mk (f : FreeMonoid (M ⊕ N) →* P) (hM₁ hN₁ hM hN w) :
    clift f hM₁ hN₁ hM hN (mk w) = f w :=
  rfl


@[to_additive (attr := simp)]
theorem clift_comp_mk (f : FreeMonoid (M ⊕ N) →* P) (hM₁ hN₁ hM hN) :
    (clift f hM₁ hN₁ hM hN).comp mk = f :=
  DFunLike.ext' rfl


@[to_additive (attr := simp)]
theorem mclosure_range_inl_union_inr :
    Submonoid.closure (range (inl : M →* M ∗ N) ∪ range (inr : N →* M ∗ N)) = ⊤ := by
  rw [← mrange_mk, MonoidHom.mrange_eq_map, ← closure_range_of, MonoidHom.map_mclosure,
                                 /-
                                   M : Type u_1
                                   N : Type u_2
                                   inst✝¹ : MulOneClass M
                                   inst✝ : MulOneClass N
                                   ⊢ Eq (Submonoid.closure (Union.union (Set.range ⇑Monoid.Coprod.inl) (Set.range …
                                 -/
    ← range_comp, Sum.range_eq]; rfl
                                 /-
                                   🎉 no goals
                                 -/


@[to_additive (attr := simp)] theorem mrange_inl_sup_mrange_inr :
    MonoidHom.mrange (inl : M →* M ∗ N) ⊔ MonoidHom.mrange (inr : N →* M ∗ N) = ⊤ := by
  rw [← mclosure_range_inl_union_inr, Submonoid.closure_union, ← MonoidHom.coe_mrange,
    ← MonoidHom.coe_mrange, Submonoid.closure_eq, Submonoid.closure_eq]


@[to_additive]
theorem codisjoint_mrange_inl_mrange_inr :
    Codisjoint (MonoidHom.mrange (inl : M →* M ∗ N)) (MonoidHom.mrange inr) :=
  codisjoint_iff.2 mrange_inl_sup_mrange_inr


@[to_additive] theorem mrange_eq (f : M ∗ N →* P) :
    MonoidHom.mrange f = MonoidHom.mrange (f.comp inl) ⊔ MonoidHom.mrange (f.comp inr) := by
  rw [MonoidHom.mrange_eq_map, ← mrange_inl_sup_mrange_inr, Submonoid.map_sup, MonoidHom.map_mrange,
    MonoidHom.map_mrange]


/-- Extensionality lemma for monoid homomorphisms `M ∗ N →* P`.
If two homomorphisms agree on the ranges of `Monoid.Coprod.inl` and `Monoid.Coprod.inr`,
then they are equal. -/
@[to_additive (attr := ext 1100)
  "Extensionality lemma for additive monoid homomorphisms `AddMonoid.Coprod M N →+ P`.
  If two homomorphisms agree on the ranges of `AddMonoid.Coprod.inl` and `AddMonoid.Coprod.inr`,
  then they are equal."]
theorem hom_ext {f g : M ∗ N →* P} (h₁ : f.comp inl = g.comp inl) (h₂ : f.comp inr = g.comp inr) :
    f = g :=
  MonoidHom.eq_of_eqOn_denseM mclosure_range_inl_union_inr <| eqOn_union.2
    ⟨eqOn_range.2 <| DFunLike.ext'_iff.1 h₁, eqOn_range.2 <| DFunLike.ext'_iff.1 h₂⟩


@[to_additive (attr := simp)]
theorem clift_mk :
    clift (mk : FreeMonoid (M ⊕ N) →* M ∗ N) (map_one inl) (map_one inr) (map_mul inl)
      (map_mul inr) = .id _ :=
  hom_ext rfl rfl


/-- Map `M ∗ N` to `M' ∗ N'` by applying `Sum.map f g` to each element of the underlying list. -/
@[to_additive "Map `AddMonoid.Coprod M N` to `AddMonoid.Coprod M' N'`
by applying `Sum.map f g` to each element of the underlying list."]
def map (f : M →* M') (g : N →* N') : M ∗ N →* M' ∗ N' :=
  clift (mk.comp <| FreeMonoid.map <| Sum.map f g)
        /-
          M : Type u_1
          N : Type u_2
          M' : Type u_3
          N' : Type u_4
          P : Type u_5
          inst✝⁴ : MulOneClass M
          inst✝³ : MulOneClass N
          inst✝² : MulOneClass M'
          inst✝¹ : MulOneClass N'
          inst✝ : MulOneClass P
          f : MonoidHom M M'
          g : MonoidHom N N'
          ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map (Sum.map ⇑f ⇑g))) (FreeMonoid.of  …
        -/
    (by simp only [MonoidHom.comp_apply, map_of, Sum.map_inl, map_one, mk_of_inl])
        /-
          🎉 no goals
        -/
        /-
          M : Type u_1
          N : Type u_2
          M' : Type u_3
          N' : Type u_4
          P : Type u_5
          inst✝⁴ : MulOneClass M
          inst✝³ : MulOneClass N
          inst✝² : MulOneClass M'
          inst✝¹ : MulOneClass N'
          inst✝ : MulOneClass P
          f : MonoidHom M M'
          g : MonoidHom N N'
          ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map (Sum.map ⇑f ⇑g))) (FreeMonoid.of  …
        -/
    (by simp only [MonoidHom.comp_apply, map_of, Sum.map_inr, map_one, mk_of_inr])
        /-
          🎉 no goals
        -/
                   /-
                     M : Type u_1
                     N : Type u_2
                     M' : Type u_3
                     N' : Type u_4
                     P : Type u_5
                     inst✝⁴ : MulOneClass M
                     inst✝³ : MulOneClass N
                     inst✝² : MulOneClass M'
                     inst✝¹ : MulOneClass N'
                     inst✝ : MulOneClass P
                     f : MonoidHom M M'
                     g : MonoidHom N N'
                     x y : M
                     ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map (Sum.map ⇑f ⇑g))) (FreeMonoid.of  …
                   -/
    (fun x y => by simp only [MonoidHom.comp_apply, map_of, Sum.map_inl, map_mul, mk_of_inl])
                   /-
                     🎉 no goals
                   -/
                  /-
                    M : Type u_1
                    N : Type u_2
                    M' : Type u_3
                    N' : Type u_4
                    P : Type u_5
                    inst✝⁴ : MulOneClass M
                    inst✝³ : MulOneClass N
                    inst✝² : MulOneClass M'
                    inst✝¹ : MulOneClass N'
                    inst✝ : MulOneClass P
                    f : MonoidHom M M'
                    g : MonoidHom N N'
                    x y : N
                    ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map (Sum.map ⇑f ⇑g))) (FreeMonoid.of  …
                  -/
    fun x y => by simp only [MonoidHom.comp_apply, map_of, Sum.map_inr, map_mul, mk_of_inr]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem map_mk_ofList (f : M →* M') (g : N →* N') (l : List (M ⊕ N)) :
    map f g (mk (ofList l)) = mk (ofList (l.map (Sum.map f g))) :=
  rfl


@[to_additive (attr := simp)]
theorem map_apply_inl (f : M →* M') (g : N →* N') (x : M) : map f g (inl x) = inl (f x) := rfl


@[to_additive (attr := simp)]
theorem map_apply_inr (f : M →* M') (g : N →* N') (x : N) : map f g (inr x) = inr (g x) := rfl


@[to_additive (attr := simp)]
theorem map_comp_inl (f : M →* M') (g : N →* N') : (map f g).comp inl = inl.comp f := rfl


@[to_additive (attr := simp)]
theorem map_comp_inr (f : M →* M') (g : N →* N') : (map f g).comp inr = inr.comp g := rfl


@[to_additive (attr := simp)]
theorem map_id_id : map (.id M) (.id N) = .id (M ∗ N) := hom_ext rfl rfl


@[to_additive]
theorem map_comp_map {M'' N''} [MulOneClass M''] [MulOneClass N''] (f' : M' →* M'') (g' : N' →* N'')
    (f : M →* M') (g : N →* N') : (map f' g').comp (map f g) = map (f'.comp f) (g'.comp g) :=
  hom_ext rfl rfl


@[to_additive]
theorem map_map {M'' N''} [MulOneClass M''] [MulOneClass N''] (f' : M' →* M'') (g' : N' →* N'')
    (f : M →* M') (g : N →* N') (x : M ∗ N) :
    map f' g' (map f g x) = map (f'.comp f) (g'.comp g) x :=
  DFunLike.congr_fun (map_comp_map f' g' f g) x


/-- Map `M ∗ N` to `N ∗ M` by applying `Sum.swap` to each element of the underlying list.

See also `MulEquiv.coprodComm` for a `MulEquiv` version. -/
@[to_additive "Map `AddMonoid.Coprod M N` to `AddMonoid.Coprod N M`
  by applying `Sum.swap` to each element of the underlying list.

See also `AddEquiv.coprodComm` for an `AddEquiv` version."]
def swap : M ∗ N →* N ∗ M :=
  clift (mk.comp <| FreeMonoid.map Sum.swap)
        /-
          M : Type u_1
          N : Type u_2
          M' : Type u_3
          N' : Type u_4
          P : Type u_5
          inst✝⁴ : MulOneClass M
          inst✝³ : MulOneClass N
          inst✝² : MulOneClass M'
          inst✝¹ : MulOneClass N'
          inst✝ : MulOneClass P
          ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map Sum.swap)) (FreeMonoid.of (Sum.in …
        -/
    (by simp only [MonoidHom.comp_apply, map_of, Sum.swap_inl, mk_of_inr, map_one])
        /-
          🎉 no goals
        -/
        /-
          M : Type u_1
          N : Type u_2
          M' : Type u_3
          N' : Type u_4
          P : Type u_5
          inst✝⁴ : MulOneClass M
          inst✝³ : MulOneClass N
          inst✝² : MulOneClass M'
          inst✝¹ : MulOneClass N'
          inst✝ : MulOneClass P
          ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map Sum.swap)) (FreeMonoid.of (Sum.in …
        -/
    (by simp only [MonoidHom.comp_apply, map_of, Sum.swap_inr, mk_of_inl, map_one])
        /-
          🎉 no goals
        -/
                   /-
                     M : Type u_1
                     N : Type u_2
                     M' : Type u_3
                     N' : Type u_4
                     P : Type u_5
                     inst✝⁴ : MulOneClass M
                     inst✝³ : MulOneClass N
                     inst✝² : MulOneClass M'
                     inst✝¹ : MulOneClass N'
                     inst✝ : MulOneClass P
                     x y : M
                     ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map Sum.swap)) (FreeMonoid.of (Sum.in …
                   -/
    (fun x y => by simp only [MonoidHom.comp_apply, map_of, Sum.swap_inl, mk_of_inr, map_mul])
                   /-
                     🎉 no goals
                   -/
                   /-
                     M : Type u_1
                     N : Type u_2
                     M' : Type u_3
                     N' : Type u_4
                     P : Type u_5
                     inst✝⁴ : MulOneClass M
                     inst✝³ : MulOneClass N
                     inst✝² : MulOneClass M'
                     inst✝¹ : MulOneClass N'
                     inst✝ : MulOneClass P
                     x y : N
                     ⊢ Eq ((Monoid.Coprod.mk.comp (FreeMonoid.map Sum.swap)) (FreeMonoid.of (Sum.in …
                   -/
    (fun x y => by simp only [MonoidHom.comp_apply, map_of, Sum.swap_inr, mk_of_inl, map_mul])
                   /-
                     🎉 no goals
                   -/


@[to_additive (attr := simp)]
theorem swap_comp_swap : (swap M N).comp (swap N M) = .id _ := hom_ext rfl rfl


@[to_additive (attr := simp)]
theorem swap_swap (x : M ∗ N) : swap N M (swap M N x) = x :=
  DFunLike.congr_fun (swap_comp_swap _ _) x


@[to_additive]
theorem swap_comp_map (f : M →* M') (g : N →* N') :
    (swap M' N').comp (map f g) = (map g f).comp (swap M N) :=
  hom_ext rfl rfl


@[to_additive]
theorem swap_map (f : M →* M') (g : N →* N') (x : M ∗ N) :
    swap M' N' (map f g x) = map g f (swap M N x) :=
  DFunLike.congr_fun (swap_comp_map f g) x


@[to_additive (attr := simp)] theorem swap_comp_inl : (swap M N).comp inl = inr := rfl

@[to_additive (attr := simp)] theorem swap_inl (x : M) : swap M N (inl x) = inr x := rfl

@[to_additive (attr := simp)] theorem swap_comp_inr : (swap M N).comp inr = inl := rfl

@[to_additive (attr := simp)] theorem swap_inr (x : N) : swap M N (inr x) = inl x := rfl


@[to_additive]
theorem swap_injective : Injective (swap M N) := LeftInverse.injective swap_swap


@[to_additive (attr := simp)]
theorem swap_inj {x y : M ∗ N} : swap M N x = swap M N y ↔ x = y := swap_injective.eq_iff


@[to_additive (attr := simp)]
theorem swap_eq_one {x : M ∗ N} : swap M N x = 1 ↔ x = 1 := swap_injective.eq_iff' (map_one _)


@[to_additive]
theorem swap_surjective : Surjective (swap M N) := LeftInverse.surjective swap_swap


@[to_additive]
theorem swap_bijective : Bijective (swap M N) := ⟨swap_injective, swap_surjective⟩


@[to_additive (attr := simp)]
theorem mker_swap : MonoidHom.mker (swap M N) = ⊥ := Submonoid.ext fun _ ↦ swap_eq_one


@[to_additive (attr := simp)]
theorem mrange_swap : MonoidHom.mrange (swap M N) = ⊤ :=
  MonoidHom.mrange_eq_top_of_surjective _ swap_surjective


/-- Lift a pair of monoid homomorphisms `f : M →* P`, `g : N →* P`
to a monoid homomorphism `M ∗ N →* P`.

See also `Coprod.clift` for a version that allows custom computational behavior
and works for a `MulOneClass` codomain.
-/
@[to_additive "Lift a pair of additive monoid homomorphisms `f : M →+ P`, `g : N →+ P`
to an additive monoid homomorphism `AddMonoid.Coprod M N →+ P`.

See also `AddMonoid.Coprod.clift` for a version that allows custom computational behavior
and works for an `AddZeroClass` codomain."]
def lift (f : M →* P) (g : N →* P) : (M ∗ N) →* P :=
  clift (FreeMonoid.lift <| Sum.elim f g) (map_one f) (map_one g) (map_mul f) (map_mul g)


@[to_additive (attr := simp)]
theorem lift_apply_mk (f : M →* P) (g : N →* P) (x : FreeMonoid (M ⊕ N)) :
    lift f g (mk x) = FreeMonoid.lift (Sum.elim f g) x :=
  rfl


@[to_additive (attr := simp)]
theorem lift_apply_inl (f : M →* P) (g : N →* P) (x : M) : lift f g (inl x) = f x :=
  rfl


@[to_additive]
theorem lift_unique {f : M →* P} {g : N →* P} {fg : M ∗ N →* P} (h₁ : fg.comp inl = f)
    (h₂ : fg.comp inr = g) : fg = lift f g :=
  hom_ext h₁ h₂


@[to_additive (attr := simp)]
theorem lift_comp_inl (f : M →* P) (g : N →* P) : (lift f g).comp inl = f := rfl


@[to_additive (attr := simp)]
theorem lift_apply_inr (f : M →* P) (g : N →* P) (x : N) : lift f g (inr x) = g x :=
  rfl


@[to_additive (attr := simp)]
theorem lift_comp_inr (f : M →* P) (g : N →* P) : (lift f g).comp inr = g := rfl


@[to_additive (attr := simp)]
theorem lift_comp_swap (f : M →* P) (g : N →* P) : (lift f g).comp (swap N M) = lift g f :=
  hom_ext rfl rfl


@[to_additive (attr := simp)]
theorem lift_swap (f : M →* P) (g : N →* P) (x : N ∗ M) : lift f g (swap N M x) = lift g f x :=
  DFunLike.congr_fun (lift_comp_swap f g) x


@[to_additive]
theorem comp_lift {P' : Type*} [Monoid P'] (f : P →* P') (g₁ : M →* P) (g₂ : N →* P) :
    f.comp (lift g₁ g₂) = lift (f.comp g₁) (f.comp g₂) :=
              /-
                M : Type u_1
                N : Type u_2
                P : Type u_3
                inst✝³ : MulOneClass M
                inst✝² : MulOneClass N
                inst✝¹ : Monoid P
                P' : Type u_4
                inst✝ : Monoid P'
                f : MonoidHom P P'
                g₁ : MonoidHom M P
                g₂ : MonoidHom N P
                ⊢ Eq ((f.comp (Monoid.Coprod.lift g₁ g₂)).comp Monoid.Coprod.inl) ((Monoid.Cop …
              -/
  hom_ext (by rw [MonoidHom.comp_assoc, lift_comp_inl, lift_comp_inl]) <| by
              /-
                🎉 no goals
              -/
    /-
      M : Type u_1
      N : Type u_2
      P : Type u_3
      inst✝³ : MulOneClass M
      inst✝² : MulOneClass N
      inst✝¹ : Monoid P
      P' : Type u_4
      inst✝ : Monoid P'
      f : MonoidHom P P'
      g₁ : MonoidHom M P
      g₂ : MonoidHom N P
      ⊢ Eq ((f.comp (Monoid.Coprod.lift g₁ g₂)).comp Monoid.Coprod.inr) ((Monoid.Cop …
    -/
    rw [MonoidHom.comp_assoc, lift_comp_inr, lift_comp_inr]
    /-
      🎉 no goals
    -/


/-- `Coprod.lift` as an equivalence. -/
@[to_additive "`AddMonoid.Coprod.lift` as an equivalence."]
def liftEquiv : (M →* P) × (N →* P) ≃ (M ∗ N →* P) where
  toFun fg := lift fg.1 fg.2
  invFun f := (f.comp inl, f.comp inr)
  left_inv _ := rfl
  right_inv _ := Eq.symm <| lift_unique rfl rfl


@[to_additive (attr := simp)]
theorem mrange_lift (f : M →* P) (g : N →* P) :
    MonoidHom.mrange (lift f g) = MonoidHom.mrange f ⊔ MonoidHom.mrange g := by
  /-
    M : Type u_1
    N : Type u_2
    P : Type u_3
    inst✝² : MulOneClass M
    inst✝¹ : MulOneClass N
    inst✝ : Monoid P
    f : MonoidHom M P
    g : MonoidHom N P
    ⊢ Eq (MonoidHom.mrange (Monoid.Coprod.lift f g)) (Max.max (MonoidHom.mrange f) …
  -/
  simp [mrange_eq]
  /-
    🎉 no goals
  -/


@[to_additive] instance : Monoid (M ∗ N) :=
  { mul_assoc := (Con.monoid _).mul_assoc
    one_mul := (Con.monoid _).one_mul
    mul_one := (Con.monoid _).mul_one }


/-- The natural projection `M ∗ N →* M`. -/
@[to_additive "The natural projection `AddMonoid.Coprod M N →+ M`."]
def fst : M ∗ N →* M := lift (.id M) 1


/-- The natural projection `M ∗ N →* N`. -/
@[to_additive "The natural projection `AddMonoid.Coprod M N →+ N`."]
def snd : M ∗ N →* N := lift 1 (.id N)


/-- The natural projection `M ∗ N →* M × N`. -/
@[to_additive "The natural projection `AddMonoid.Coprod M N →+ M × N`."]
def toProd : M ∗ N →* M × N := lift (.inl _ _) (.inr _ _)


@[to_additive (attr := simp)] theorem fst_comp_inl : (fst : M ∗ N →* M).comp inl = .id _ := rfl

@[to_additive (attr := simp)] theorem fst_apply_inl (x : M) : fst (inl x : M ∗ N) = x := rfl

@[to_additive (attr := simp)] theorem fst_comp_inr : (fst : M ∗ N →* M).comp inr = 1 := rfl

@[to_additive (attr := simp)] theorem fst_apply_inr (x : N) : fst (inr x : M ∗ N) = 1 := rfl

@[to_additive (attr := simp)] theorem snd_comp_inl : (snd : M ∗ N →* N).comp inl = 1 := rfl

@[to_additive (attr := simp)] theorem snd_apply_inl (x : M) : snd (inl x : M ∗ N) = 1 := rfl

@[to_additive (attr := simp)] theorem snd_comp_inr : (snd : M ∗ N →* N).comp inr = .id _ := rfl

@[to_additive (attr := simp)] theorem snd_apply_inr (x : N) : snd (inr x : M ∗ N) = x := rfl


@[to_additive (attr := simp)]
theorem toProd_comp_inl : (toProd : M ∗ N →* M × N).comp inl = .inl _ _ := rfl


@[to_additive (attr := simp)]
theorem toProd_comp_inr : (toProd : M ∗ N →* M × N).comp inr = .inr _ _ := rfl


@[to_additive (attr := simp)]
theorem toProd_apply_inl (x : M) : toProd (inl x : M ∗ N) = (x, 1) := rfl


@[to_additive (attr := simp)]
theorem toProd_apply_inr (x : N) : toProd (inr x : M ∗ N) = (1, x) := rfl


@[to_additive (attr := simp)]
                                                                  /-
                                                                    M : Type u_1
                                                                    N : Type u_2
                                                                    inst✝¹ : Monoid M
                                                                    inst✝ : Monoid N
                                                                    ⊢ Eq (Monoid.Coprod.fst.prod Monoid.Coprod.snd) Monoid.Coprod.toProd
                                                                  -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
theorem fst_prod_snd : (fst : M ∗ N →* M).prod snd = toProd := by ext1 <;> rfl
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[to_additive (attr := simp)]
theorem prod_mk_fst_snd (x : M ∗ N) : (fst x, snd x) = toProd x := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : Monoid M
    inst✝ : Monoid N
    x : Monoid.Coprod M N
    ⊢ Eq { fst := Monoid.Coprod.fst x, snd := Monoid.Coprod.snd x } (Monoid.Coprod …
  -/
  rw [← fst_prod_snd, MonoidHom.prod_apply]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem fst_comp_toProd : (MonoidHom.fst M N).comp toProd = fst := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : Monoid M
    inst✝ : Monoid N
    ⊢ Eq ((MonoidHom.fst M N).comp Monoid.Coprod.toProd) Monoid.Coprod.fst
  -/
  rw [← fst_prod_snd, MonoidHom.fst_comp_prod]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem fst_toProd (x : M ∗ N) : (toProd x).1 = fst x := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : Monoid M
    inst✝ : Monoid N
    x : Monoid.Coprod M N
    ⊢ Eq (Monoid.Coprod.toProd x).1 (Monoid.Coprod.fst x)
  -/
  rw [← fst_comp_toProd]; rfl
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp)]
theorem snd_comp_toProd : (MonoidHom.snd M N).comp toProd = snd := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : Monoid M
    inst✝ : Monoid N
    ⊢ Eq ((MonoidHom.snd M N).comp Monoid.Coprod.toProd) Monoid.Coprod.snd
  -/
  rw [← fst_prod_snd, MonoidHom.snd_comp_prod]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem snd_toProd (x : M ∗ N) : (toProd x).2 = snd x := by
  /-
    M : Type u_1
    N : Type u_2
    inst✝¹ : Monoid M
    inst✝ : Monoid N
    x : Monoid.Coprod M N
    ⊢ Eq (Monoid.Coprod.toProd x).2 (Monoid.Coprod.snd x)
  -/
  rw [← snd_comp_toProd]; rfl
                          /-
                            🎉 no goals
                          -/


@[to_additive (attr := simp)]
theorem fst_comp_swap : fst.comp (swap M N) = snd := lift_comp_swap _ _


@[to_additive (attr := simp)]
theorem fst_swap (x : M ∗ N) : fst (swap M N x) = snd x := lift_swap _ _ _


@[to_additive (attr := simp)]
theorem snd_comp_swap : snd.comp (swap M N) = fst := lift_comp_swap _ _


@[to_additive (attr := simp)]
theorem snd_swap (x : M ∗ N) : snd (swap M N x) = fst x := lift_swap _ _ _


@[to_additive (attr := simp)]
theorem lift_inr_inl : lift (inr : M →* N ∗ M) inl = swap M N := hom_ext rfl rfl


@[to_additive (attr := simp)]
theorem lift_inl_inr : lift (inl : M →* M ∗ N) inr = .id _ := hom_ext rfl rfl


@[to_additive]
theorem inl_injective : Injective (inl : M →* M ∗ N) := LeftInverse.injective fst_apply_inl


@[to_additive]
theorem inr_injective : Injective (inr : N →* M ∗ N) := LeftInverse.injective snd_apply_inr


@[to_additive]
theorem fst_surjective : Surjective (fst : M ∗ N →* M) := LeftInverse.surjective fst_apply_inl


@[to_additive]
theorem snd_surjective : Surjective (snd : M ∗ N →* N) := LeftInverse.surjective snd_apply_inr


@[to_additive]
theorem toProd_surjective : Surjective (toProd : M ∗ N →* M × N) := fun x =>
                         /-
                           M : Type u_1
                           N : Type u_2
                           inst✝¹ : Monoid M
                           inst✝ : Monoid N
                           x : Prod M N
                           ⊢ Eq (Monoid.Coprod.toProd (HMul.hMul (Monoid.Coprod.inl x.1) (Monoid.Coprod.i …
                         -/
  ⟨inl x.1 * inr x.2, by rw [map_mul, toProd_apply_inl, toProd_apply_inr, Prod.fst_mul_snd]⟩
                         /-
                           🎉 no goals
                         -/


@[to_additive]
theorem mk_of_inv_mul : ∀ x : G ⊕ H, mk (of (x.map Inv.inv Inv.inv)) * mk (of x) = 1
  | Sum.inl _ => map_mul_eq_one inl (inv_mul_cancel _)
  | Sum.inr _ => map_mul_eq_one inr (inv_mul_cancel _)


@[to_additive]
theorem con_inv_mul_cancel (x : FreeMonoid (G ⊕ H)) :
    coprodCon G H (ofList (x.toList.map (Sum.map Inv.inv Inv.inv)).reverse * x) 1 := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝¹ : Group G
    inst✝ : Group H
    x : FreeMonoid (Sum G H)
    ⊢ (Monoid.coprodCon G H) (HMul.hMul (FreeMonoid.ofList (List.map (Sum.map Inv. …
  -/
  rw [← mk_eq_mk, map_mul, map_one]
  induction x using FreeMonoid.inductionOn' with
  | one => simp
  | mul_of x xs ihx =>
    simp only [toList_of_mul, map_cons, reverse_cons, ofList_append, map_mul, ihx, ofList_singleton]
    rwa [mul_assoc, ← mul_assoc (mk (of _)), mk_of_inv_mul, one_mul]


@[to_additive]
instance : Inv (G ∗ H) where
  inv := Quotient.map' (fun w => ofList (w.toList.map (Sum.map Inv.inv Inv.inv)).reverse) fun _ _ ↦
    (coprodCon G H).map_of_mul_left_rel_one _ con_inv_mul_cancel


@[to_additive]
theorem inv_def (w : FreeMonoid (G ⊕ H)) :
    (mk w)⁻¹ = mk (ofList (w.toList.map (Sum.map Inv.inv Inv.inv)).reverse) :=
  rfl


@[to_additive]
instance : Group (G ∗ H) where
  inv_mul_cancel := mk_surjective.forall.2 fun x => mk_eq_mk.2 (con_inv_mul_cancel x)


@[to_additive (attr := simp)]
theorem closure_range_inl_union_inr :
    Subgroup.closure (range (inl : G →* G ∗ H) ∪ range inr) = ⊤ :=
  Subgroup.closure_eq_top_of_mclosure_eq_top mclosure_range_inl_union_inr


@[to_additive (attr := simp)] theorem range_inl_sup_range_inr :
    MonoidHom.range (inl : G →* G ∗ H) ⊔ MonoidHom.range inr = ⊤ := by
  rw [← closure_range_inl_union_inr, Subgroup.closure_union, ← MonoidHom.coe_range,
    ← MonoidHom.coe_range, Subgroup.closure_eq, Subgroup.closure_eq]


@[to_additive]
theorem codisjoint_range_inl_range_inr :
    Codisjoint (MonoidHom.range (inl : G →* G ∗ H)) (MonoidHom.range inr) :=
  codisjoint_iff.2 range_inl_sup_range_inr


@[to_additive (attr := simp)] theorem range_swap : MonoidHom.range (swap G H) = ⊤ :=
  MonoidHom.range_eq_top.2 swap_surjective


@[to_additive] theorem range_eq (f : G ∗ H →* K) :
    MonoidHom.range f = MonoidHom.range (f.comp inl) ⊔ MonoidHom.range (f.comp inr) := by
  rw [MonoidHom.range_eq_map, ← range_inl_sup_range_inr, Subgroup.map_sup, MonoidHom.map_range,
    MonoidHom.map_range]


@[to_additive (attr := simp)] theorem range_lift (f : G →* K) (g : H →* K) :
    MonoidHom.range (lift f g) = MonoidHom.range f ⊔ MonoidHom.range g := by
  /-
    G : Type u_1
    H : Type u_2
    inst✝² : Group G
    inst✝¹ : Group H
    K : Type u_3
    inst✝ : Group K
    f : MonoidHom G K
    g : MonoidHom H K
    ⊢ Eq (Monoid.Coprod.lift f g).range (Max.max f.range g.range)
  -/
  simp [range_eq]
  /-
    🎉 no goals
  -/


/-- Lift two monoid equivalences `e : M ≃* N` and `e' : M' ≃* N'` to a monoid equivalence
`(M ∗ M') ≃* (N ∗ N')`. -/
@[to_additive (attr := simps! (config := .asFn)) "Lift two additive monoid
equivalences `e : M ≃+ N` and `e' : M' ≃+ N'` to an additive monoid equivalence
`(AddMonoid.Coprod M M') ≃+ (AddMonoid.Coprod N N')`."]
def coprodCongr (e : M ≃* N) (e' : M' ≃* N') : (M ∗ M') ≃* (N ∗ N') :=
  (Coprod.map (e : M →* N) (e' : M' →* N')).toMulEquiv (Coprod.map e.symm e'.symm)
        /-
          M : Type u_1
          N : Type u_2
          M' : Type u_3
          N' : Type u_4
          inst✝³ : MulOneClass M
          inst✝² : MulOneClass N
          inst✝¹ : MulOneClass M'
          inst✝ : MulOneClass N'
          e : MulEquiv M N
          e' : MulEquiv M' N'
          ⊢ Eq ((Monoid.Coprod.map ↑e.symm ↑e'.symm).comp (Monoid.Coprod.map ↑e ↑e')) (M …
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
    (by ext <;> simp) (by ext <;> simp)
                                  /-
                                    🎉 no goals
                                  -/


/-- A `MulEquiv` version of `Coprod.swap`. -/
@[to_additive (attr := simps! (config := .asFn))
  "An `AddEquiv` version of `AddMonoid.Coprod.swap`."]
def coprodComm : M ∗ N ≃* N ∗ M :=
  (Coprod.swap _ _).toMulEquiv (Coprod.swap _ _) (Coprod.swap_comp_swap _ _)
    (Coprod.swap_comp_swap _ _)


/-- A multiplicative equivalence between `(M ∗ N) ∗ P` and `M ∗ (N ∗ P)`. -/
@[to_additive "An additive equivalence between `AddMonoid.Coprod (AddMonoid.Coprod M N) P` and
`AddMonoid.Coprod M (AddMonoid.Coprod N P)`."]
def coprodAssoc : (M ∗ N) ∗ P ≃* M ∗ (N ∗ P) :=
  MonoidHom.toMulEquiv
    (Coprod.lift (Coprod.map (.id M) inl) (inr.comp inr))
    (Coprod.lift (inl.comp inl) (Coprod.map inr (.id P)))
        /-
          M : Type u_1
          N : Type u_2
          P : Type u_3
          inst✝² : Monoid M
          inst✝¹ : Monoid N
          inst✝ : Monoid P
          ⊢ Eq ((Monoid.Coprod.lift (Monoid.Coprod.inl.comp Monoid.Coprod.inl) (Monoid.C …
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
                                 /-
                                   🎉 no goals
                                 -/
    (by ext <;> rfl) (by ext <;> rfl)
                                 /-
                                   🎉 no goals
                                 -/


@[to_additive (attr := simp)]
theorem coprodAssoc_apply_inl_inl (x : M) : coprodAssoc M N P (inl (inl x)) = inl x := rfl


@[to_additive (attr := simp)]
theorem coprodAssoc_apply_inl_inr (x : N) : coprodAssoc M N P (inl (inr x)) = inr (inl x) := rfl


@[to_additive (attr := simp)]
theorem coprodAssoc_apply_inr (x : P) : coprodAssoc M N P (inr x) = inr (inr x) := rfl


@[to_additive (attr := simp)]
theorem coprodAssoc_symm_apply_inl (x : M) : (coprodAssoc M N P).symm (inl x) = inl (inl x) :=
  rfl


@[to_additive (attr := simp)]
theorem coprodAssoc_symm_apply_inr_inl (x : N) :
    (coprodAssoc M N P).symm (inr (inl x)) = inl (inr x) :=
  rfl


@[to_additive (attr := simp)]
theorem coprodAssoc_symm_apply_inr_inr (x : P) :
    (coprodAssoc M N P).symm (inr (inr x)) = inr x :=
  rfl


/-- Isomorphism between `M ∗ PUnit` and `M`. -/
@[simps! (config := .asFn)]
def coprodPUnit : M ∗ PUnit ≃* M :=
  MonoidHom.toMulEquiv fst inl (hom_ext rfl <| Subsingleton.elim _ _) fst_comp_inl


/-- Isomorphism between `PUnit ∗ M` and `M`. -/
@[simps! (config := .asFn)]
def punitCoprod : PUnit ∗ M ≃* M :=
  MonoidHom.toMulEquiv snd inr (hom_ext (Subsingleton.elim _ _) rfl) snd_comp_inr


/-- Isomorphism between `M ∗ PUnit` and `M`. -/
@[simps! (config := .asFn)]
def coprodUnit : AddMonoid.Coprod M PUnit ≃+ M :=
  AddMonoidHom.toAddEquiv AddMonoid.Coprod.fst AddMonoid.Coprod.inl
    (AddMonoid.Coprod.hom_ext rfl <| Subsingleton.elim _ _) AddMonoid.Coprod.fst_comp_inl


/-- Isomorphism between `PUnit ∗ M` and `M`. -/
@[simps! (config := .asFn)]
def punitCoprod : AddMonoid.Coprod PUnit M ≃+ M :=
  AddMonoidHom.toAddEquiv AddMonoid.Coprod.snd AddMonoid.Coprod.inr
    (AddMonoid.Coprod.hom_ext (Subsingleton.elim _ _) rfl) AddMonoid.Coprod.snd_comp_inr


