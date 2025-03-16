@[to_additive]
theorem list_prod_apply {α : Type*} {β : α → Type*} [∀ a, Monoid (β a)] (a : α)
    (l : List (∀ a, β a)) : l.prod a = (l.map fun f : ∀ a, β a ↦ f a).prod :=
  map_list_prod (evalMonoidHom β a) _


@[to_additive]
theorem multiset_prod_apply {α : Type*} {β : α → Type*} [∀ a, CommMonoid (β a)] (a : α)
    (s : Multiset (∀ a, β a)) : s.prod a = (s.map fun f : ∀ a, β a ↦ f a).prod :=
  (evalMonoidHom β a).map_multiset_prod _


@[to_additive (attr := simp)]
theorem Finset.prod_apply {α : Type*} {β : α → Type*} {γ} [∀ a, CommMonoid (β a)] (a : α)
    (s : Finset γ) (g : γ → ∀ a, β a) : (∏ c ∈ s, g c) a = ∏ c ∈ s, g c a :=
  map_prod (Pi.evalMonoidHom β a) _ _


/-- An 'unapplied' analogue of `Finset.prod_apply`. -/
@[to_additive "An 'unapplied' analogue of `Finset.sum_apply`."]
theorem Finset.prod_fn {α : Type*} {β : α → Type*} {γ} [∀ a, CommMonoid (β a)] (s : Finset γ)
    (g : γ → ∀ a, β a) : ∏ c ∈ s, g c = fun a ↦ ∏ c ∈ s, g c a :=
  funext fun _ ↦ Finset.prod_apply _ _ _


@[to_additive]
theorem Fintype.prod_apply {α : Type*} {β : α → Type*} {γ : Type*} [Fintype γ]
    [∀ a, CommMonoid (β a)] (a : α) (g : γ → ∀ a, β a) : (∏ c, g c) a = ∏ c, g c a :=
  Finset.prod_apply a Finset.univ g


@[to_additive prod_mk_sum]
theorem prod_mk_prod {α β γ : Type*} [CommMonoid α] [CommMonoid β] (s : Finset γ) (f : γ → α)
    (g : γ → β) : (∏ x ∈ s, f x, ∏ x ∈ s, g x) = ∏ x ∈ s, (f x, g x) :=
  haveI := Classical.decEq γ
                                /-
                                  α : Type u_4
                                  β : Type u_5
                                  γ : Type u_6
                                  inst✝¹ : CommMonoid α
                                  inst✝ : CommMonoid β
                                  s : Finset γ
                                  f : γ → α
                                  g : γ → β
                                  this : DecidableEq γ
                                  ⊢ ∀ ⦃a : γ⦄ {s : Finset γ}, Not (Membership.mem s a) → Eq { fst := s.prod fun  …
                                -/
  Finset.induction_on s rfl (by simp +contextual [Prod.ext_iff])
                                /-
                                  🎉 no goals
                                -/


/-- decomposing `x : ι → R` as a sum along the canonical basis -/
theorem pi_eq_sum_univ {ι : Type*} [Fintype ι] [DecidableEq ι] {R : Type*} [Semiring R]
    (x : ι → R) : x = ∑ i, (x i) • fun j => if i = j then (1 : R) else 0 := by
  /-
    ι : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    R : Type u_5
    inst✝ : Semiring R
    x : ι → R
    ⊢ Eq x (Finset.univ.sum fun i => HSMul.hSMul (x i) fun j => ite (Eq i j) 1 0)
  -/
  ext
  /-
    case h
    ι : Type u_4
    inst✝² : Fintype ι
    inst✝¹ : DecidableEq ι
    R : Type u_5
    inst✝ : Semiring R
    x : ι → R
    x✝ : ι
    ⊢ Eq (x x✝) (Finset.univ.sum (fun i => HSMul.hSMul (x i) fun j => ite (Eq i j) …
  -/
  simp
  /-
    🎉 no goals
  -/


lemma prod_indicator_apply (s : Finset ι) (f : ι → Set κ) (g : ι → κ → α) (j : κ) :
    ∏ i ∈ s, (f i).indicator (g i) j = (⋂ x ∈ s, f x).indicator (∏ i ∈ s, g i) j := by
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝ : CommSemiring α
    s : Finset ι
    f : ι → Set κ
    g : ι → κ → α
    j : κ
    ⊢ Eq (s.prod fun i => (f i).indicator (g i) j) ((Set.iInter fun x => Set.iInte …
  -/
  rw [Set.indicator]
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝ : CommSemiring α
    s : Finset ι
    f : ι → Set κ
    g : ι → κ → α
    j : κ
    ⊢ Eq (s.prod fun i => (f i).indicator (g i) j) (ite (Membership.mem (Set.iInte …
  -/
  split_ifs with hj
    /-
      case pos
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝ : CommSemiring α
      s : Finset ι
      f : ι → Set κ
      g : ι → κ → α
      j : κ
      hj : Membership.mem (Set.iInter fun x => Set.iInter fun h => f x) j
      ⊢ Eq (s.prod fun i => (f i).indicator (g i) j) (s.prod (fun i => g i) j)
    -/
  · rw [Finset.prod_apply]
    /-
      case pos
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝ : CommSemiring α
      s : Finset ι
      f : ι → Set κ
      g : ι → κ → α
      j : κ
      hj : Membership.mem (Set.iInter fun x => Set.iInter fun h => f x) j
      ⊢ Eq (s.prod fun i => (f i).indicator (g i) j) (s.prod fun c => g c j)
    -/
    congr! 1 with i hi
    /-
      case pos.a
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝ : CommSemiring α
      s : Finset ι
      f : ι → Set κ
      g : ι → κ → α
      j : κ
      hj : Membership.mem (Set.iInter fun x => Set.iInter fun h => f x) j
      i : ι
      hi : Membership.mem s i
      ⊢ Eq ((f i).indicator (g i) j) (g i j)
    -/
    simp only [Finset.inf_set_eq_iInter, Set.mem_iInter] at hj
    /-
      case pos.a
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝ : CommSemiring α
      s : Finset ι
      f : ι → Set κ
      g : ι → κ → α
      j : κ
      i : ι
      hi : Membership.mem s i
      hj : ∀ (i : ι), Membership.mem s i → Membership.mem (f i) j
      ⊢ Eq ((f i).indicator (g i) j) (g i j)
    -/
    exact Set.indicator_of_mem (hj _ hi) _
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝ : CommSemiring α
      s : Finset ι
      f : ι → Set κ
      g : ι → κ → α
      j : κ
      hj : Not (Membership.mem (Set.iInter fun x => Set.iInter fun h => f x) j)
      ⊢ Eq (s.prod fun i => (f i).indicator (g i) j) 0
    -/
  · obtain ⟨i, hi, hj⟩ := by simpa using hj
    /-
      case neg.intro.intro
      ι : Type u_1
      κ : Type u_2
      α : Type u_3
      inst✝ : CommSemiring α
      s : Finset ι
      f : ι → Set κ
      g : ι → κ → α
      j : κ
      hj✝ : Not (Membership.mem (Set.iInter fun x => Set.iInter fun h => f x) j)
      i : ι
      hi : Membership.mem s i
      hj : Not (Membership.mem (f i) j)
      ⊢ Eq (s.prod fun i => (f i).indicator (g i) j) 0
    -/
    exact Finset.prod_eq_zero hi <| Set.indicator_of_not_mem hj _
    /-
      🎉 no goals
    -/


lemma prod_indicator (s : Finset ι) (f : ι → Set κ) (g : ι → κ → α) :
    ∏ i ∈ s, (f i).indicator (g i) = (⋂ x ∈ s, f x).indicator (∏ i ∈ s, g i) := by
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝ : CommSemiring α
    s : Finset ι
    f : ι → Set κ
    g : ι → κ → α
    ⊢ Eq (s.prod fun i => (f i).indicator (g i)) ((Set.iInter fun x => Set.iInter  …
  -/
  ext a; simpa using prod_indicator_apply ..
         /-
           🎉 no goals
         -/


lemma prod_indicator_const_apply (s : Finset ι) (f : ι → Set κ) (g : κ → α) (j : κ) :
    ∏ i ∈ s, (f i).indicator g j = (⋂ x ∈ s, f x).indicator (g ^ #s) j := by
  /-
    ι : Type u_1
    κ : Type u_2
    α : Type u_3
    inst✝ : CommSemiring α
    s : Finset ι
    f : ι → Set κ
    g : κ → α
    j : κ
    ⊢ Eq (s.prod fun i => (f i).indicator g j) ((Set.iInter fun x => Set.iInter fu …
  -/
  simp [prod_indicator_apply]
  /-
    🎉 no goals
  -/


lemma prod_indicator_const (s : Finset ι) (f : ι → Set κ) (g : κ → α) :
                                                                         /-
                                                                           ι : Type u_1
                                                                           κ : Type u_2
                                                                           α : Type u_3
                                                                           inst✝ : CommSemiring α
                                                                           s : Finset ι
                                                                           f : ι → Set κ
                                                                           g : κ → α
                                                                           ⊢ Eq (s.prod fun i => (f i).indicator g) ((Set.iInter fun x => Set.iInter fun  …
                                                                         -/
    ∏ i ∈ s, (f i).indicator g = (⋂ x ∈ s, f x).indicator (g ^ #s) := by simp [prod_indicator]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[to_additive]
theorem Finset.univ_prod_mulSingle [Fintype I] (f : ∀ i, Z i) :
    (∏ i, Pi.mulSingle i (f i)) = f := by
  /-
    I : Type u_4
    inst✝² : DecidableEq I
    Z : I → Type u_5
    inst✝¹ : (i : I) → CommMonoid (Z i)
    inst✝ : Fintype I
    f : (i : I) → Z i
    ⊢ Eq (Finset.univ.prod fun i => Pi.mulSingle i (f i)) f
  -/
  ext a
  /-
    case h
    I : Type u_4
    inst✝² : DecidableEq I
    Z : I → Type u_5
    inst✝¹ : (i : I) → CommMonoid (Z i)
    inst✝ : Fintype I
    f : (i : I) → Z i
    a : I
    ⊢ Eq (Finset.univ.prod (fun i => Pi.mulSingle i (f i)) a) (f a)
  -/
  simp
  /-
    🎉 no goals
  -/


@[to_additive]
theorem MonoidHom.functions_ext [Finite I] (G : Type*) [CommMonoid G] (g h : (∀ i, Z i) →* G)
    (H : ∀ i x, g (Pi.mulSingle i x) = h (Pi.mulSingle i x)) : g = h := by
  /-
    I : Type u_4
    inst✝³ : DecidableEq I
    Z : I → Type u_5
    inst✝² : (i : I) → CommMonoid (Z i)
    inst✝¹ : Finite I
    G : Type u_6
    inst✝ : CommMonoid G
    g h : MonoidHom ((i : I) → Z i) G
    H : ∀ (i : I) (x : Z i), Eq (g (Pi.mulSingle i x)) (h (Pi.mulSingle i x))
    ⊢ Eq g h
  -/
  cases nonempty_fintype I
  /-
    case intro
    I : Type u_4
    inst✝³ : DecidableEq I
    Z : I → Type u_5
    inst✝² : (i : I) → CommMonoid (Z i)
    inst✝¹ : Finite I
    G : Type u_6
    inst✝ : CommMonoid G
    g h : MonoidHom ((i : I) → Z i) G
    H : ∀ (i : I) (x : Z i), Eq (g (Pi.mulSingle i x)) (h (Pi.mulSingle i x))
    val✝ : Fintype I
    ⊢ Eq g h
  -/
  ext k
  /-
    case intro.h
    I : Type u_4
    inst✝³ : DecidableEq I
    Z : I → Type u_5
    inst✝² : (i : I) → CommMonoid (Z i)
    inst✝¹ : Finite I
    G : Type u_6
    inst✝ : CommMonoid G
    g h : MonoidHom ((i : I) → Z i) G
    H : ∀ (i : I) (x : Z i), Eq (g (Pi.mulSingle i x)) (h (Pi.mulSingle i x))
    val✝ : Fintype I
    k : (i : I) → Z i
    ⊢ Eq (g k) (h k)
  -/
  rw [← Finset.univ_prod_mulSingle k, map_prod, map_prod]
  /-
    case intro.h
    I : Type u_4
    inst✝³ : DecidableEq I
    Z : I → Type u_5
    inst✝² : (i : I) → CommMonoid (Z i)
    inst✝¹ : Finite I
    G : Type u_6
    inst✝ : CommMonoid G
    g h : MonoidHom ((i : I) → Z i) G
    H : ∀ (i : I) (x : Z i), Eq (g (Pi.mulSingle i x)) (h (Pi.mulSingle i x))
    val✝ : Fintype I
    k : (i : I) → Z i
    ⊢ Eq (Finset.univ.prod fun x => g (Pi.mulSingle x (k x))) (Finset.univ.prod fu …
  -/
  simp only [H]
  /-
    🎉 no goals
  -/


/-- This is used as the ext lemma instead of `MonoidHom.functions_ext` for reasons explained in
note [partially-applied ext lemmas]. -/
@[to_additive (attr := ext)
      "\nThis is used as the ext lemma instead of `AddMonoidHom.functions_ext` for reasons
      explained in note [partially-applied ext lemmas]."]
theorem MonoidHom.functions_ext' [Finite I] (M : Type*) [CommMonoid M] (g h : (∀ i, Z i) →* M)
    (H : ∀ i, g.comp (MonoidHom.mulSingle Z i) = h.comp (MonoidHom.mulSingle Z i)) : g = h :=
  g.functions_ext M h fun i => DFunLike.congr_fun (H i)


@[ext]
theorem RingHom.functions_ext [Finite I] (G : Type*) [NonAssocSemiring G] (g h : (∀ i, f i) →+* G)
    (H : ∀ (i : I) (x : f i), g (single i x) = h (single i x)) : g = h :=
  RingHom.coe_addMonoidHom_injective <|
    @AddMonoidHom.functions_ext I _ f _ _ G _ (g : (∀ i, f i) →+ G) h H


@[to_additive]
theorem fst_prod : (∏ c ∈ s, f c).1 = ∏ c ∈ s, (f c).1 :=
  map_prod (MonoidHom.fst α β) f s


@[to_additive]
theorem snd_prod : (∏ c ∈ s, f c).2 = ∏ c ∈ s, (f c).2 :=
  map_prod (MonoidHom.snd α β) f s


/-- The canonical isomorphism between the monoid of homomorphisms from a finite product of
commutative monoids to another commutative monoid and the product of the homomorphism monoids. -/
@[to_additive "The canonical isomorphism between the additive monoid of homomorphisms from
a finite product of additive commutative monoids to another additive commutative monoid and
the product of the homomorphism monoids."]
def Pi.monoidHomMulEquiv {ι : Type*} [Fintype ι] [DecidableEq ι] (M : ι → Type*)
    [(i : ι) → CommMonoid (M i)] (M' : Type*) [CommMonoid M'] :
    (((i : ι) → M i) →* M') ≃* ((i : ι) → (M i →* M')) where
  toFun φ i := φ.comp <| MonoidHom.mulSingle M i
  invFun φ := ∏ (i : ι), (φ i).comp (Pi.evalMonoidHom M i)
  left_inv φ := by
    /-
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : MonoidHom ((i : ι) → M i) M'
      ⊢ Eq ((fun φ => Finset.univ.prod fun i => (φ i).comp (Pi.evalMonoidHom M i)) ( …
    -/
    ext
    simp only [MonoidHom.finset_prod_apply, MonoidHom.coe_comp, Function.comp_apply,
      evalMonoidHom_apply, MonoidHom.mulSingle_apply, ← map_prod]
    /-
      case H.h
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : MonoidHom ((i : ι) → M i) M'
      i✝ : ι
      x✝ : M i✝
      ⊢ Eq (φ (Finset.univ.prod fun x => Pi.mulSingle x (Pi.mulSingle i✝ x✝ x))) (φ  …
    -/
    refine congrArg _ <| funext fun _ ↦ ?_
    /-
      case H.h
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : MonoidHom ((i : ι) → M i) M'
      i✝ : ι
      x✝¹ : M i✝
      x✝ : ι
      ⊢ Eq (Finset.univ.prod (fun x => Pi.mulSingle x (Pi.mulSingle i✝ x✝¹ x)) x✝) ( …
    -/
    rw [Fintype.prod_apply]
    /-
      case H.h
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : MonoidHom ((i : ι) → M i) M'
      i✝ : ι
      x✝¹ : M i✝
      x✝ : ι
      ⊢ Eq (Finset.univ.prod fun c => Pi.mulSingle c (Pi.mulSingle i✝ x✝¹ c) x✝) (Pi …
    -/
    exact Fintype.prod_pi_mulSingle ..
    /-
      🎉 no goals
    -/
  right_inv φ := by
    /-
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : (i : ι) → MonoidHom (M i) M'
      ⊢ Eq ((fun φ i => φ.comp (MonoidHom.mulSingle M i)) ((fun φ => Finset.univ.pro …
    -/
    ext i m
    simp only [MonoidHom.coe_comp, Function.comp_apply, MonoidHom.mulSingle_apply,
      MonoidHom.finset_prod_apply, evalMonoidHom_apply, ]
    /-
      case h.h
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : (i : ι) → MonoidHom (M i) M'
      i : ι
      m : M i
      ⊢ Eq (Finset.univ.prod fun x => (φ x) (Pi.mulSingle i m x)) ((φ i) m)
    -/
    let φ' i : M i → M' := ⇑(φ i)
    conv =>
      enter [1, 2, j]
      rw [show φ j = φ' j from rfl, Pi.apply_mulSingle φ' (fun i ↦ map_one (φ i))]
    /-
      case h.h
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : (i : ι) → MonoidHom (M i) M'
      i : ι
      m : M i
      φ' : (i : ι) → M i → M' := fun i => ⇑(φ i)
      ⊢ Eq (Finset.univ.prod (Pi.mulSingle i (φ' i m))) ((φ i) m)
    -/
    rw [show φ' i = φ i from rfl]
    /-
      case h.h
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ : (i : ι) → MonoidHom (M i) M'
      i : ι
      m : M i
      φ' : (i : ι) → M i → M' := fun i => ⇑(φ i)
      ⊢ Eq (Finset.univ.prod (Pi.mulSingle i ((φ i) m))) ((φ i) m)
    -/
    exact Fintype.prod_pi_mulSingle' ..
    /-
      🎉 no goals
    -/
  map_mul' φ ψ := by
    /-
      ι✝ : Type u_1
      κ : Type u_2
      α : Type u_3
      ι : Type u_4
      inst✝³ : Fintype ι
      inst✝² : DecidableEq ι
      M : ι → Type u_5
      inst✝¹ : (i : ι) → CommMonoid (M i)
      M' : Type u_6
      inst✝ : CommMonoid M'
      φ ψ : MonoidHom ((i : ι) → M i) M'
      ⊢ Eq ({ toFun := fun φ i => φ.comp (MonoidHom.mulSingle M i), invFun := fun φ  …
    -/
    ext
    simp only [MonoidHom.coe_comp, Function.comp_apply, MonoidHom.mulSingle_apply,
      MonoidHom.mul_apply, mul_apply]


