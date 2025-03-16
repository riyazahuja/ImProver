/-- `FreeCommRing α` is the free commutative ring on the type `α`. -/
def FreeCommRing (α : Type u) : Type u :=
  FreeAbelianGroup <| Multiplicative <| Multiset α

-- Porting note: two instances below couldn't be derived

instance FreeCommRing.instCommRing : CommRing (FreeCommRing α) := by
  /-
    α : Type u
    ⊢ CommRing (FreeCommRing α)
  -/
  delta FreeCommRing; infer_instance
                      /-
                        🎉 no goals
                      -/


instance FreeCommRing.instInhabited : Inhabited (FreeCommRing α) := by
  /-
    α : Type u
    ⊢ Inhabited (FreeCommRing α)
  -/
  delta FreeCommRing; infer_instance
                      /-
                        🎉 no goals
                      -/


/-- The canonical map from `α` to the free commutative ring on `α`. -/
def of (x : α) : FreeCommRing α :=
  FreeAbelianGroup.of <| Multiplicative.ofAdd ({x} : Multiset α)


theorem of_injective : Function.Injective (of : α → FreeCommRing α) :=
  FreeAbelianGroup.of_injective.comp fun _ _ =>
    (Multiset.coe_eq_coe.trans List.singleton_perm_singleton).mp


@[simp]
theorem of_ne_zero (x : α) : of x ≠ 0 := FreeAbelianGroup.of_ne_zero _


@[simp]
theorem zero_ne_of (x : α) : 0 ≠ of x := FreeAbelianGroup.zero_ne_of _


@[simp]
theorem of_ne_one (x : α) : of x ≠ 1 :=
  FreeAbelianGroup.of_injective.ne <| Multiset.singleton_ne_zero _


@[simp]
theorem one_ne_of (x : α) : 1 ≠ of x :=
  FreeAbelianGroup.of_injective.ne <| Multiset.zero_ne_singleton _

-- Porting note: added to ease a proof in `Mathlib.Algebra.Colimit.Ring`

lemma of_cons (a : α) (m : Multiset α) : (FreeAbelianGroup.of (Multiplicative.ofAdd (a ::ₘ m))) =
    @HMul.hMul _ (FreeCommRing α) (FreeCommRing α) _ (of a)
    (FreeAbelianGroup.of (Multiplicative.ofAdd m)) := by
  /-
    α : Type u
    a : α
    m : Multiset α
    ⊢ Eq (FreeAbelianGroup.of (Multiplicative.ofAdd (Multiset.cons a m))) (HMul.hM …
  -/
  dsimp [FreeCommRing]
  rw [← Multiset.singleton_add, ofAdd_add,
    of, FreeAbelianGroup.of_mul_of]


@[elab_as_elim, induction_eliminator]
protected theorem induction_on {C : FreeCommRing α → Prop} (z : FreeCommRing α) (hn1 : C (-1))
    (hb : ∀ b, C (of b)) (ha : ∀ x y, C x → C y → C (x + y)) (hm : ∀ x y, C x → C y → C (x * y)) :
    C z :=
  have hn : ∀ x, C x → C (-x) := fun x ih => neg_one_mul x ▸ hm _ _ hn1 ih
  have h1 : C 1 := neg_neg (1 : FreeCommRing α) ▸ hn _ hn1
  FreeAbelianGroup.induction_on z (neg_add_cancel (1 : FreeCommRing α) ▸ ha _ _ hn1 h1)
    (fun m => Multiset.induction_on m h1 fun a m ih => by
      /-
        α : Type u
        C : FreeCommRing α → Prop
        z : FreeCommRing α
        hn1 : C (-1)
        hb : ∀ (b : α), C (FreeCommRing.of b)
        ha : ∀ (x y : FreeCommRing α), C x → C y → C (HAdd.hAdd x y)
        hm : ∀ (x y : FreeCommRing α), C x → C y → C (HMul.hMul x y)
        hn : ∀ (x : FreeCommRing α), C x → C (Neg.neg x)
        h1 : C 1
        m✝ : Multiplicative (Multiset α)
        a : α
        m : Multiset α
        ih : C (FreeAbelianGroup.of m)
        ⊢ C (FreeAbelianGroup.of (Multiset.cons a m))
      -/
      convert hm (of a) _ (hb a) ih
      /-
        case h.e'_1
        α : Type u
        C : FreeCommRing α → Prop
        z : FreeCommRing α
        hn1 : C (-1)
        hb : ∀ (b : α), C (FreeCommRing.of b)
        ha : ∀ (x y : FreeCommRing α), C x → C y → C (HAdd.hAdd x y)
        hm : ∀ (x y : FreeCommRing α), C x → C y → C (HMul.hMul x y)
        hn : ∀ (x : FreeCommRing α), C x → C (Neg.neg x)
        h1 : C 1
        m✝ : Multiplicative (Multiset α)
        a : α
        m : Multiset α
        ih : C (FreeAbelianGroup.of m)
        ⊢ Eq (FreeAbelianGroup.of (Multiset.cons a m)) (HMul.hMul (FreeCommRing.of a)  …
      -/
      apply of_cons)
      /-
        🎉 no goals
      -/
    (fun _ ih => hn _ ih) ha


/-- A helper to implement `lift`. This is essentially `FreeCommMonoid.lift`, but this does not
currently exist. -/
private def liftToMultiset : (α → R) ≃ (Multiplicative (Multiset α) →* R) where
  toFun f :=
    { toFun := fun s => (s.toAdd.map f).prod
      map_mul' := fun x y =>
        calc
          _ = Multiset.prod (Multiset.map f x + Multiset.map f y) := by
            /-
              α : Type u
              R : Type v
              inst✝ : CommRing R
              f✝ f : α → R
              x y : Multiplicative (Multiset α)
              ⊢ Eq ({ toFun := fun s => (Multiset.map f (Multiplicative.toAdd s)).prod, map_ …
            -/
            rw [← Multiset.map_add]
            /-
              α : Type u
              R : Type v
              inst✝ : CommRing R
              f✝ f : α → R
              x y : Multiplicative (Multiset α)
              ⊢ Eq ({ toFun := fun s => (Multiset.map f (Multiplicative.toAdd s)).prod, map_ …
            -/
            rfl
            /-
              🎉 no goals
            -/
          _ = _ := Multiset.prod_add _ _
      map_one' := rfl }
  invFun F x := F (Multiplicative.ofAdd ({x} : Multiset α))
                                                                      /-
                                                                        α : Type u
                                                                        R : Type v
                                                                        inst✝ : CommRing R
                                                                        f✝ f : α → R
                                                                        x : α
                                                                        ⊢ Eq (Multiset.map f (Singleton.singleton x)).prod (f x)
                                                                      -/
  left_inv f := funext fun x => show (Multiset.map f {x}).prod = _ by simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
  right_inv F := MonoidHom.ext fun x =>
    let F' := MonoidHom.toAdditive'' F
    let x' := x.toAdd
    show (Multiset.map (fun a => F' {a}) x').sum = F' x' by
      /-
        α : Type u
        R : Type v
        inst✝ : CommRing R
        f : α → R
        F : MonoidHom (Multiplicative (Multiset α)) R
        x : Multiplicative (Multiset α)
        F' : AddMonoidHom (Multiset α) (Additive R) := MonoidHom.toAdditive'' F
        x' : Multiset α := Multiplicative.toAdd x
        ⊢ Eq (Multiset.map (fun a => F' (Singleton.singleton a)) x').sum (F' x')
      -/
      erw [← Multiset.map_map (fun x => F' x) (fun x => {x}), ← AddMonoidHom.map_multiset_sum]
      /-
        α : Type u
        R : Type v
        inst✝ : CommRing R
        f : α → R
        F : MonoidHom (Multiplicative (Multiset α)) R
        x : Multiplicative (Multiset α)
        F' : AddMonoidHom (Multiset α) (Additive R) := MonoidHom.toAdditive'' F
        x' : Multiset α := Multiplicative.toAdd x
        ⊢ Eq (F' (Multiset.map (fun x => Singleton.singleton x) x').sum) (F' x')
      -/
      exact DFunLike.congr_arg F (Multiset.sum_map_singleton x')
      /-
        🎉 no goals
      -/


/-- Lift a map `α → R` to an additive group homomorphism `FreeCommRing α → R`. -/
def lift : (α → R) ≃ (FreeCommRing α →+* R) :=
  Equiv.trans liftToMultiset FreeAbelianGroup.liftMonoid


@[simp]
theorem lift_of (x : α) : lift f (of x) = f x :=
  (FreeAbelianGroup.lift.of _ _).trans <| mul_one _


@[simp]
theorem lift_comp_of (f : FreeCommRing α →+* R) : lift (f ∘ of) = f :=
  RingHom.ext fun x =>
                                    /-
                                      α : Type u
                                      R : Type v
                                      inst✝ : CommRing R
                                      f : RingHom (FreeCommRing α) R
                                      x : FreeCommRing α
                                      ⊢ Eq ((FreeCommRing.lift (Function.comp (⇑f) FreeCommRing.of)) (-1)) (f (-1))
                                    -/
    FreeCommRing.induction_on x (by rw [RingHom.map_neg, RingHom.map_one, f.map_neg, f.map_one])
                                    /-
                                      🎉 no goals
                                    -/
                                         /-
                                           α : Type u
                                           R : Type v
                                           inst✝ : CommRing R
                                           f : RingHom (FreeCommRing α) R
                                           x✝ x y : FreeCommRing α
                                           ihx : Eq ((FreeCommRing.lift (Function.comp (⇑f) FreeCommRing.of)) x) (f x)
                                           ihy : Eq ((FreeCommRing.lift (Function.comp (⇑f) FreeCommRing.of)) y) (f y)
                                           ⊢ Eq ((FreeCommRing.lift (Function.comp (⇑f) FreeCommRing.of)) (HAdd.hAdd x y) …
                                         -/
      (lift_of _) (fun x y ihx ihy => by rw [RingHom.map_add, f.map_add, ihx, ihy])
                                         /-
                                           🎉 no goals
                                         -/
                            /-
                              α : Type u
                              R : Type v
                              inst✝ : CommRing R
                              f : RingHom (FreeCommRing α) R
                              x✝ x y : FreeCommRing α
                              ihx : Eq ((FreeCommRing.lift (Function.comp (⇑f) FreeCommRing.of)) x) (f x)
                              ihy : Eq ((FreeCommRing.lift (Function.comp (⇑f) FreeCommRing.of)) y) (f y)
                              ⊢ Eq ((FreeCommRing.lift (Function.comp (⇑f) FreeCommRing.of)) (HMul.hMul x y) …
                            -/
      fun x y ihx ihy => by rw [RingHom.map_mul, f.map_mul, ihx, ihy]
                            /-
                              🎉 no goals
                            -/


@[ext 1100]
theorem hom_ext ⦃f g : FreeCommRing α →+* R⦄ (h : ∀ x, f (of x) = g (of x)) : f = g :=
  lift.symm.injective (funext h)


/-- A map `f : α → β` produces a ring homomorphism `FreeCommRing α →+* FreeCommRing β`. -/
def map : FreeCommRing α →+* FreeCommRing β :=
  lift <| of ∘ f


@[simp]
theorem map_of (x : α) : map f (of x) = of (f x) :=
  lift_of _ _


/-- `is_supported x s` means that all monomials showing up in `x` have variables in `s`. -/
def IsSupported (x : FreeCommRing α) (s : Set α) : Prop :=
  x ∈ Subring.closure (of '' s)


theorem isSupported_upwards (hs : IsSupported x s) (hst : s ⊆ t) : IsSupported x t :=
  Subring.closure_mono (Set.monotone_image hst) hs


theorem isSupported_add (hxs : IsSupported x s) (hys : IsSupported y s) : IsSupported (x + y) s :=
  Subring.add_mem _ hxs hys


theorem isSupported_neg (hxs : IsSupported x s) : IsSupported (-x) s :=
  Subring.neg_mem _ hxs


theorem isSupported_sub (hxs : IsSupported x s) (hys : IsSupported y s) : IsSupported (x - y) s :=
  Subring.sub_mem _ hxs hys


theorem isSupported_mul (hxs : IsSupported x s) (hys : IsSupported y s) : IsSupported (x * y) s :=
  Subring.mul_mem _ hxs hys


theorem isSupported_zero : IsSupported 0 s :=
  Subring.zero_mem _


theorem isSupported_one : IsSupported 1 s :=
  Subring.one_mem _


theorem isSupported_int {i : ℤ} {s : Set α} : IsSupported (↑i) s :=
  Int.induction_on i isSupported_zero
                    /-
                      α : Type u
                      i✝ : Int
                      s : Set α
                      i : Nat
                      hi : (↑↑i).IsSupported s
                      ⊢ (↑(HAdd.hAdd (↑i) 1)).IsSupported s
                    -/
    (fun i hi => by rw [Int.cast_add, Int.cast_one]; exact isSupported_add hi isSupported_one)
                                                     /-
                                                       🎉 no goals
                                                     -/
                   /-
                     α : Type u
                     i✝ : Int
                     s : Set α
                     i : Nat
                     hi : (↑(Neg.neg ↑i)).IsSupported s
                     ⊢ (↑(HSub.hSub (Neg.neg ↑i) 1)).IsSupported s
                   -/
    fun i hi => by rw [Int.cast_sub, Int.cast_one]; exact isSupported_sub hi isSupported_one
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- The restriction map from `FreeCommRing α` to `FreeCommRing s` where `s : Set α`, defined
  by sending all variables not in `s` to zero. -/
def restriction (s : Set α) [DecidablePred (· ∈ s)] : FreeCommRing α →+* FreeCommRing s :=
  lift (fun a => if H : a ∈ s then of ⟨a, H⟩ else 0)


@[simp]
theorem restriction_of (p) : restriction s (of p) = if H : p ∈ s then of ⟨p, H⟩ else 0 :=
  lift_of _ _


theorem isSupported_of {p} {s : Set α} : IsSupported (of p) s ↔ p ∈ s :=
  suffices IsSupported (of p) s → p ∈ s from ⟨this, fun hps => Subring.subset_closure ⟨p, hps, rfl⟩⟩
  fun hps : IsSupported (of p) s => by
  /-
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    ⊢ Membership.mem s p
  -/
  haveI := Classical.decPred s
  have : ∀ x, IsSupported x s →
        ∃ n : ℤ, lift (fun a => if a ∈ s then (0 : ℤ[X]) else Polynomial.X) x = n := by
    intro x hx
    refine Subring.InClosure.recOn hx ?_ ?_ ?_ ?_
    · use 1
      rw [RingHom.map_one]
      norm_cast
    · use -1
      rw [RingHom.map_neg, RingHom.map_one, Int.cast_neg, Int.cast_one]
    · rintro _ ⟨z, hzs, rfl⟩ _ _
      use 0
      rw [RingHom.map_mul, lift_of, if_pos hzs, zero_mul]
      norm_cast
    · rintro x y ⟨q, hq⟩ ⟨r, hr⟩
      refine ⟨q + r, ?_⟩
      rw [RingHom.map_add, hq, hr]
      norm_cast
  /-
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this✝ : DecidablePred s
    this : ∀ (x : FreeCommRing α), x.IsSupported s → Exists fun n => Eq ((FreeComm …
    ⊢ Membership.mem s p
  -/
  specialize this (of p) hps
  /-
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this✝ : DecidablePred s
    this : Exists fun n => Eq ((FreeCommRing.lift fun a => ite (Membership.mem s a …
    ⊢ Membership.mem s p
  -/
  rw [lift_of] at this
  /-
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this✝ : DecidablePred s
    this : Exists fun n => Eq (ite (Membership.mem s p) 0 Polynomial.X) ↑n
    ⊢ Membership.mem s p
  -/
  split_ifs at this with h
    /-
      case pos
      α : Type u
      p : α
      s : Set α
      hps : (FreeCommRing.of p).IsSupported s
      this✝ : DecidablePred s
      h : Membership.mem s p
      this : Exists fun n => Eq 0 ↑n
      ⊢ Membership.mem s p
    -/
  · exact h
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this✝ : DecidablePred s
    h : Not (Membership.mem s p)
    this : Exists fun n => Eq Polynomial.X ↑n
    ⊢ Membership.mem s p
  -/
  exfalso
  /-
    case neg
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this✝ : DecidablePred s
    h : Not (Membership.mem s p)
    this : Exists fun n => Eq Polynomial.X ↑n
    ⊢ False
  -/
  apply Ne.symm Int.zero_ne_one
  /-
    case neg
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this✝ : DecidablePred s
    h : Not (Membership.mem s p)
    this : Exists fun n => Eq Polynomial.X ↑n
    ⊢ Eq 1 0
  -/
  rcases this with ⟨w, H⟩
  /-
    case neg.intro
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this : DecidablePred s
    h : Not (Membership.mem s p)
    w : Int
    H : Eq Polynomial.X ↑w
    ⊢ Eq 1 0
  -/
  rw [← Polynomial.C_eq_intCast] at H
  /-
    case neg.intro
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this : DecidablePred s
    h : Not (Membership.mem s p)
    w : Int
    H : Eq Polynomial.X (Polynomial.C ↑w)
    ⊢ Eq 1 0
  -/
  have : Polynomial.X.coeff 1 = (Polynomial.C ↑w).coeff 1 := by rw [H]; rfl
  /-
    case neg.intro
    α : Type u
    p : α
    s : Set α
    hps : (FreeCommRing.of p).IsSupported s
    this✝ : DecidablePred s
    h : Not (Membership.mem s p)
    w : Int
    H : Eq Polynomial.X (Polynomial.C ↑w)
    this : Eq (Polynomial.X.coeff 1) ((Polynomial.C w).coeff 1)
    ⊢ Eq 1 0
  -/
  rwa [Polynomial.coeff_C, if_neg (one_ne_zero : 1 ≠ 0), Polynomial.coeff_X, if_pos rfl] at this
  /-
    🎉 no goals
  -/

-- Porting note: Changed `(Subtype.val : s → α)` to `(↑)` in the type

theorem map_subtype_val_restriction {x} (s : Set α) [DecidablePred (· ∈ s)]
    (hxs : IsSupported x s) : map (↑) (restriction s x) = x := by
  /-
    α : Type u
    x : FreeCommRing α
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    hxs : x.IsSupported s
    ⊢ Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) x)) x
  -/
  refine Subring.InClosure.recOn hxs ?_ ?_ ?_ ?_
    /-
      case refine_1
      α : Type u
      x : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x.IsSupported s
      ⊢ Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) 1)) 1
    -/
  · rw [RingHom.map_one]
    /-
      case refine_1
      α : Type u
      x : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x.IsSupported s
      ⊢ Eq ((FreeCommRing.map Subtype.val) 1) 1
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      x : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x.IsSupported s
      ⊢ Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) (-1))) (-1)
    -/
  · rw [map_neg, map_one]
    /-
      case refine_2
      α : Type u
      x : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x.IsSupported s
      ⊢ Eq ((FreeCommRing.map Subtype.val) (-1)) (-1)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u
      x : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x.IsSupported s
      ⊢ ∀ (z : FreeCommRing α), Membership.mem (Set.image FreeCommRing.of s) z → ∀ ( …
    -/
  · rintro _ ⟨p, hps, rfl⟩ n ih
    /-
      case refine_3.intro.intro
      α : Type u
      x : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x.IsSupported s
      p : α
      hps : Membership.mem s p
      n : FreeCommRing α
      ih : Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) n)) n
      ⊢ Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) (HMul.hMul  …
    -/
    rw [RingHom.map_mul, restriction_of, dif_pos hps, RingHom.map_mul, map_of, ih]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u
      x : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x.IsSupported s
      ⊢ ∀ {x y : FreeCommRing α}, Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing. …
    -/
  · intro x y ihx ihy
    /-
      case refine_4
      α : Type u
      x✝ : FreeCommRing α
      s : Set α
      inst✝ : DecidablePred fun x => Membership.mem s x
      hxs : x✝.IsSupported s
      x y : FreeCommRing α
      ihx : Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) x)) x
      ihy : Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) y)) y
      ⊢ Eq ((FreeCommRing.map Subtype.val) ((FreeCommRing.restriction s) (HAdd.hAdd  …
    -/
    rw [RingHom.map_add, RingHom.map_add, ihx, ihy]
    /-
      🎉 no goals
    -/


theorem exists_finite_support (x : FreeCommRing α) : ∃ s : Set α, Set.Finite s ∧ IsSupported x s :=
  FreeCommRing.induction_on x ⟨∅, Set.finite_empty, isSupported_neg isSupported_one⟩
    (fun p => ⟨{p}, Set.finite_singleton p, isSupported_of.2 <| Set.mem_singleton _⟩)
    (fun _ _ ⟨s, hfs, hxs⟩ ⟨t, hft, hxt⟩ =>
      ⟨s ∪ t, hfs.union hft,
        isSupported_add (isSupported_upwards hxs Set.subset_union_left)
          (isSupported_upwards hxt Set.subset_union_right)⟩)
    fun _ _ ⟨s, hfs, hxs⟩ ⟨t, hft, hxt⟩ =>
    ⟨s ∪ t, hfs.union hft,
      isSupported_mul (isSupported_upwards hxs Set.subset_union_left)
        (isSupported_upwards hxt Set.subset_union_right)⟩


theorem exists_finset_support (x : FreeCommRing α) : ∃ s : Finset α, IsSupported x ↑s :=
  let ⟨s, hfs, hxs⟩ := exists_finite_support x
                    /-
                      α : Type u
                      x : FreeCommRing α
                      s : Set α
                      hfs : s.Finite
                      hxs : x.IsSupported s
                      ⊢ x.IsSupported ↑hfs.toFinset
                    -/
  ⟨hfs.toFinset, by rwa [Set.Finite.coe_toFinset]⟩
                    /-
                      🎉 no goals
                    -/


/-- The canonical ring homomorphism from the free ring generated by `α` to the free commutative ring
    generated by `α`. -/
def toFreeCommRing {α} : FreeRing α →+* FreeCommRing α :=
  FreeRing.lift FreeCommRing.of


/-- The coercion defined by the canonical ring homomorphism from the free ring generated by `α` to
the free commutative ring generated by `α`. -/
@[coe] def castFreeCommRing {α} : FreeRing α → FreeCommRing α := toFreeCommRing


instance FreeCommRing.instCoe : Coe (FreeRing α) (FreeCommRing α) :=
  ⟨castFreeCommRing⟩


/-- The natural map `FreeRing α → FreeCommRing α`, as a `RingHom`. -/
def coeRingHom : FreeRing α →+* FreeCommRing α :=
  toFreeCommRing


@[simp, norm_cast]
protected theorem coe_zero : ↑(0 : FreeRing α) = (0 : FreeCommRing α) := rfl


@[simp, norm_cast]
protected theorem coe_one : ↑(1 : FreeRing α) = (1 : FreeCommRing α) := rfl


@[simp]
protected theorem coe_of (a : α) : ↑(FreeRing.of a) = FreeCommRing.of a :=
  FreeRing.lift_of _ _


@[simp, norm_cast]
protected theorem coe_neg (x : FreeRing α) : ↑(-x) = -(x : FreeCommRing α) := by
  /-
    α : Type u
    x : FreeRing α
    ⊢ Eq (↑(Neg.neg x)) (Neg.neg ↑x)
  -/
  rw [castFreeCommRing, map_neg]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
protected theorem coe_add (x y : FreeRing α) : ↑(x + y) = (x : FreeCommRing α) + y :=
  (FreeRing.lift _).map_add _ _


@[simp, norm_cast]
protected theorem coe_sub (x y : FreeRing α) : ↑(x - y) = (x : FreeCommRing α) - y := by
  /-
    α : Type u
    x y : FreeRing α
    ⊢ Eq (↑(HSub.hSub x y)) (HSub.hSub ↑x ↑y)
  -/
  rw [castFreeCommRing, map_sub]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
protected theorem coe_mul (x y : FreeRing α) : ↑(x * y) = (x : FreeCommRing α) * y :=
  (FreeRing.lift _).map_mul _ _


protected theorem coe_surjective : Surjective ((↑) : FreeRing α → FreeCommRing α) := fun x => by
  induction x with
  | hn1 =>
    use -1
    rfl
  | hb b =>
    exact ⟨FreeRing.of b, rfl⟩
  | ha _ _ hx hy =>
    rcases hx with ⟨x, rfl⟩; rcases hy with ⟨y, rfl⟩
    exact ⟨x + y, (FreeRing.lift _).map_add _ _⟩
  | hm _ _ hx hy =>
    rcases hx with ⟨x, rfl⟩; rcases hy with ⟨y, rfl⟩
    exact ⟨x * y, (FreeRing.lift _).map_mul _ _⟩


theorem coe_eq : ((↑) : FreeRing α → FreeCommRing α) =
    @Functor.map FreeAbelianGroup _ _ _ fun l : List α => (l : Multiset α) := by
  /-
    α : Type u
    ⊢ Eq FreeRing.castFreeCommRing (Functor.map fun l => ↑l)
  -/
  funext x
  erw [castFreeCommRing, toFreeCommRing, FreeRing.lift, Equiv.coe_trans, Function.comp,
    FreeAbelianGroup.liftMonoid_coe (FreeMonoid.lift FreeCommRing.of)]
  /-
    case h
    α : Type u
    x : FreeRing α
    ⊢ Eq ((FreeAbelianGroup.lift ⇑(FreeMonoid.lift FreeCommRing.of)) x) (Functor.m …
  -/
  dsimp [Functor.map]
  /-
    case h
    α : Type u
    x : FreeRing α
    ⊢ Eq ((FreeAbelianGroup.lift ⇑(FreeMonoid.lift FreeCommRing.of)) x) ((FreeAbel …
  -/
  rw [← AddMonoidHom.coe_coe]
  /-
    case h
    α : Type u
    x : FreeRing α
    ⊢ Eq (↑(FreeAbelianGroup.lift ⇑(FreeMonoid.lift FreeCommRing.of)) x) ((FreeAbe …
  -/
  apply FreeAbelianGroup.lift.unique; intro L
  /-
    case h.hg
    α : Type u
    x : FreeRing α
    L : FreeMonoid α
    ⊢ Eq (↑(FreeAbelianGroup.lift ⇑(FreeMonoid.lift FreeCommRing.of)) (FreeAbelian …
  -/
  erw [FreeAbelianGroup.lift.of, Function.comp]
  exact
    FreeMonoid.recOn L rfl fun hd tl ih => by
      rw [(FreeMonoid.lift _).map_mul, FreeMonoid.lift_eval_of, ih]
      conv_lhs => reduce
      rfl


/-- If α has size at most 1 then the natural map from the free ring on `α` to the
    free commutative ring on `α` is an isomorphism of rings. -/
def subsingletonEquivFreeCommRing [Subsingleton α] : FreeRing α ≃+* FreeCommRing α :=
  RingEquiv.ofBijective (coeRingHom _) (by
    have : (coeRingHom _ : FreeRing α → FreeCommRing α) =
        Functor.mapEquiv FreeAbelianGroup (Multiset.subsingletonEquiv α) :=
      coe_eq α
    /-
      α : Type u
      inst✝ : Subsingleton α
      this : Eq ⇑(FreeRing.coeRingHom α) ⇑(Functor.mapEquiv FreeAbelianGroup (Multis …
      ⊢ Function.Bijective ⇑(FreeRing.coeRingHom α)
    -/
    rw [this]
    /-
      α : Type u
      inst✝ : Subsingleton α
      this : Eq ⇑(FreeRing.coeRingHom α) ⇑(Functor.mapEquiv FreeAbelianGroup (Multis …
      ⊢ Function.Bijective ⇑(Functor.mapEquiv FreeAbelianGroup (Multiset.subsingleto …
    -/
    apply Equiv.bijective)
    /-
      🎉 no goals
    -/


instance instCommRing [Subsingleton α] : CommRing (FreeRing α) :=
  { inferInstanceAs (Ring (FreeRing α)) with
    mul_comm := fun x y => by
      rw [← (subsingletonEquivFreeCommRing α).symm_apply_apply (y * x),
        (subsingletonEquivFreeCommRing α).map_mul, mul_comm,
        ← (subsingletonEquivFreeCommRing α).map_mul,
        (subsingletonEquivFreeCommRing α).symm_apply_apply] }


/-- The free commutative ring on `α` is isomorphic to the polynomial ring over ℤ with
    variables in `α` -/
def freeCommRingEquivMvPolynomialInt : FreeCommRing α ≃+* MvPolynomial α ℤ :=
  RingEquiv.ofHomInv (FreeCommRing.lift <| (fun a => MvPolynomial.X a : α → MvPolynomial α ℤ))
    (MvPolynomial.eval₂Hom (Int.castRingHom (FreeCommRing α)) FreeCommRing.of)
        /-
          α : Type u
          ⊢ Eq ((↑(MvPolynomial.eval₂Hom (Int.castRingHom (FreeCommRing α)) FreeCommRing …
        -/
             /-
               🎉 no goals
             -/
                               /-
                                 🎉 no goals
                               -/
    (by ext; simp) (by ext <;> simp)
                               /-
                                 🎉 no goals
                               -/


/-- The free commutative ring on the empty type is isomorphic to `ℤ`. -/
def freeCommRingPemptyEquivInt : FreeCommRing PEmpty.{u + 1} ≃+* ℤ :=
  RingEquiv.trans (freeCommRingEquivMvPolynomialInt _) (MvPolynomial.isEmptyRingEquiv _ PEmpty)


/-- The free commutative ring on a type with one term is isomorphic to `ℤ[X]`. -/
def freeCommRingPunitEquivPolynomialInt : FreeCommRing PUnit.{u + 1} ≃+* ℤ[X] :=
  (freeCommRingEquivMvPolynomialInt _).trans (MvPolynomial.pUnitAlgEquiv ℤ).toRingEquiv


/-- The free ring on the empty type is isomorphic to `ℤ`. -/
def freeRingPemptyEquivInt : FreeRing PEmpty.{u + 1} ≃+* ℤ :=
  RingEquiv.trans (subsingletonEquivFreeCommRing _) freeCommRingPemptyEquivInt


/-- The free ring on a type with one term is isomorphic to `ℤ[X]`. -/
def freeRingPunitEquivPolynomialInt : FreeRing PUnit.{u + 1} ≃+* ℤ[X] :=
  RingEquiv.trans (subsingletonEquivFreeCommRing _) freeCommRingPunitEquivPolynomialInt

