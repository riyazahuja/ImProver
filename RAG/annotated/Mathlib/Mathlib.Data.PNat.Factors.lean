/-- The type of multisets of prime numbers.  Unique factorization
 gives an equivalence between this set and ℕ+, as we will formalize
 below. -/
def PrimeMultiset :=
  Multiset Nat.Primes deriving Inhabited, CanonicallyOrderedAddCommMonoid, DistribLattice,
  SemilatticeSup, Sub


instance : OrderBot PrimeMultiset where
               /-
                 ⊢ ∀ (a : PrimeMultiset), LE.le Bot.bot a
               -/
  bot_le := by simp only [bot_le, forall_const]
               /-
                 🎉 no goals
               -/


instance : OrderedSub PrimeMultiset where
  tsub_le_iff_right _ _ _ := Multiset.sub_le_iff_le_add


                                           /-
                                             ⊢ Repr PrimeMultiset
                                           -/
unsafe instance : Repr PrimeMultiset := by delta PrimeMultiset; infer_instance
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The multiset consisting of a single prime -/
def ofPrime (p : Nat.Primes) : PrimeMultiset :=
  ({p} : Multiset Nat.Primes)


theorem card_ofPrime (p : Nat.Primes) : Multiset.card (ofPrime p) = 1 :=
  rfl


/-- We can forget the primality property and regard a multiset
 of primes as just a multiset of positive integers, or a multiset
 of natural numbers.  In the opposite direction, if we have a
 multiset of positive integers or natural numbers, together with
 a proof that all the elements are prime, then we can regard it
 as a multiset of primes.  The next block of results records
 obvious properties of these coercions.
-/
def toNatMultiset : PrimeMultiset → Multiset ℕ := fun v => v.map Coe.coe


instance coeNat : Coe PrimeMultiset (Multiset ℕ) :=
  ⟨toNatMultiset⟩


/-- `PrimeMultiset.coe`, the coercion from a multiset of primes to a multiset of
naturals, promoted to an `AddMonoidHom`. -/
def coeNatMonoidHom : PrimeMultiset →+ Multiset ℕ :=
  { Multiset.mapAddMonoidHom Coe.coe with toFun := Coe.coe }


@[simp]
theorem coe_coeNatMonoidHom : (coeNatMonoidHom : PrimeMultiset → Multiset ℕ) = Coe.coe :=
  rfl


theorem coeNat_injective : Function.Injective (Coe.coe : PrimeMultiset → Multiset ℕ) :=
  Multiset.map_injective Nat.Primes.coe_nat_injective


theorem coeNat_ofPrime (p : Nat.Primes) : (ofPrime p : Multiset ℕ) = {(p : ℕ)} :=
  rfl


theorem coeNat_prime (v : PrimeMultiset) (p : ℕ) (h : p ∈ (v : Multiset ℕ)) : p.Prime := by
  /-
    v : PrimeMultiset
    p : Nat
    h : Membership.mem v.toNatMultiset p
    ⊢ Nat.Prime p
  -/
  rcases Multiset.mem_map.mp h with ⟨⟨_, hp'⟩, ⟨_, h_eq⟩⟩
  /-
    case intro.mk.intro
    v : PrimeMultiset
    p : Nat
    h : Membership.mem v.toNatMultiset p
    val✝ : Nat
    hp' : Nat.Prime val✝
    left✝ : Membership.mem v ⟨val✝, hp'⟩
    h_eq : Eq (Coe.coe ⟨val✝, hp'⟩) p
    ⊢ Nat.Prime p
  -/
  exact h_eq ▸ hp'
  /-
    🎉 no goals
  -/


/-- Converts a `PrimeMultiset` to a `Multiset ℕ+`. -/
def toPNatMultiset : PrimeMultiset → Multiset ℕ+ := fun v => v.map Coe.coe


instance coePNat : Coe PrimeMultiset (Multiset ℕ+) :=
  ⟨toPNatMultiset⟩


/-- `coePNat`, the coercion from a multiset of primes to a multiset of positive
naturals, regarded as an `AddMonoidHom`. -/
def coePNatMonoidHom : PrimeMultiset →+ Multiset ℕ+ :=
  { Multiset.mapAddMonoidHom Coe.coe with toFun := Coe.coe }


@[simp]
theorem coe_coePNatMonoidHom : (coePNatMonoidHom : PrimeMultiset → Multiset ℕ+) = Coe.coe :=
  rfl


theorem coePNat_injective : Function.Injective (Coe.coe : PrimeMultiset → Multiset ℕ+) :=
  Multiset.map_injective Nat.Primes.coe_pnat_injective


theorem coePNat_ofPrime (p : Nat.Primes) : (ofPrime p : Multiset ℕ+) = {(p : ℕ+)} :=
  rfl


theorem coePNat_prime (v : PrimeMultiset) (p : ℕ+) (h : p ∈ (v : Multiset ℕ+)) : p.Prime := by
  /-
    v : PrimeMultiset
    p : PNat
    h : Membership.mem v.toPNatMultiset p
    ⊢ p.Prime
  -/
  rcases Multiset.mem_map.mp h with ⟨⟨_, hp'⟩, ⟨_, h_eq⟩⟩
  /-
    case intro.mk.intro
    v : PrimeMultiset
    p : PNat
    h : Membership.mem v.toPNatMultiset p
    val✝ : Nat
    hp' : Nat.Prime val✝
    left✝ : Membership.mem v ⟨val✝, hp'⟩
    h_eq : Eq (Coe.coe ⟨val✝, hp'⟩) p
    ⊢ p.Prime
  -/
  exact h_eq ▸ hp'
  /-
    🎉 no goals
  -/


instance coeMultisetPNatNat : Coe (Multiset ℕ+) (Multiset ℕ) :=
  ⟨fun v => v.map Coe.coe⟩


theorem coePNat_nat (v : PrimeMultiset) : ((v : Multiset ℕ+) : Multiset ℕ) = (v : Multiset ℕ) := by
  /-
    v : PrimeMultiset
    ⊢ Eq (Multiset.map PNat.val v.toPNatMultiset) v.toNatMultiset
  -/
  change (v.map (Coe.coe : Nat.Primes → ℕ+)).map Subtype.val = v.map Subtype.val
  /-
    v : PrimeMultiset
    ⊢ Eq (Multiset.map Subtype.val (Multiset.map Coe.coe v)) (Multiset.map Subtype …
  -/
  rw [Multiset.map_map]
  /-
    v : PrimeMultiset
    ⊢ Eq (Multiset.map (Function.comp Subtype.val Coe.coe) v) (Multiset.map Subtyp …
  -/
  congr
  /-
    🎉 no goals
  -/


/-- The product of a `PrimeMultiset`, as a `ℕ+`. -/
def prod (v : PrimeMultiset) : ℕ+ :=
  (v : Multiset PNat).prod


theorem coe_prod (v : PrimeMultiset) : (v.prod : ℕ) = (v : Multiset ℕ).prod := by
  let h : (v.prod : ℕ) = ((v.map Coe.coe).map Coe.coe).prod :=
    PNat.coeMonoidHom.map_multiset_prod v.toPNatMultiset
  /-
    v : PrimeMultiset
    h : Eq (↑v.prod) (Multiset.map Coe.coe (Multiset.map Coe.coe v)).prod := Monoi …
    ⊢ Eq (↑v.prod) v.toNatMultiset.prod
  -/
  rw [Multiset.map_map] at h
  /-
    v : PrimeMultiset
    h : Eq (↑v.prod) (Multiset.map (Function.comp Coe.coe Coe.coe) v).prod
    ⊢ Eq (↑v.prod) v.toNatMultiset.prod
  -/
  have : (Coe.coe : ℕ+ → ℕ) ∘ (Coe.coe : Nat.Primes → ℕ+) = Coe.coe := funext fun p => rfl
  /-
    v : PrimeMultiset
    h : Eq (↑v.prod) (Multiset.map (Function.comp Coe.coe Coe.coe) v).prod
    this : Eq (Function.comp Coe.coe Coe.coe) Coe.coe
    ⊢ Eq (↑v.prod) v.toNatMultiset.prod
  -/
  rw [this] at h; exact h
                  /-
                    🎉 no goals
                  -/


theorem prod_ofPrime (p : Nat.Primes) : (ofPrime p).prod = (p : ℕ+) :=
  Multiset.prod_singleton _


/-- If a `Multiset ℕ` consists only of primes, it can be recast as a `PrimeMultiset`. -/
def ofNatMultiset (v : Multiset ℕ) (h : ∀ p : ℕ, p ∈ v → p.Prime) : PrimeMultiset :=
  @Multiset.pmap ℕ Nat.Primes Nat.Prime (fun p hp => ⟨p, hp⟩) v h


theorem to_ofNatMultiset (v : Multiset ℕ) (h) : (ofNatMultiset v h : Multiset ℕ) = v := by
  /-
    v : Multiset Nat
    h : ∀ (p : Nat), Membership.mem v p → Nat.Prime p
    ⊢ Eq (PrimeMultiset.ofNatMultiset v h).toNatMultiset v
  -/
  dsimp [ofNatMultiset, toNatMultiset]
  have : (fun p h => (Coe.coe : Nat.Primes → ℕ) ⟨p, h⟩) = fun p _ => id p := by
    funext p h
    rfl
  /-
    v : Multiset Nat
    h : ∀ (p : Nat), Membership.mem v p → Nat.Prime p
    this : Eq (fun p h => Coe.coe ⟨p, h⟩) fun p x => id p
    ⊢ Eq (Multiset.map Coe.coe (Multiset.pmap (fun p hp => ⟨p, hp⟩) v h)) v
  -/
  rw [Multiset.map_pmap, this, Multiset.pmap_eq_map, Multiset.map_id]
  /-
    🎉 no goals
  -/


theorem prod_ofNatMultiset (v : Multiset ℕ) (h) :
                                                        /-
                                                          v : Multiset Nat
                                                          h : ∀ (p : Nat), Membership.mem v p → Nat.Prime p
                                                          ⊢ Eq (↑(PrimeMultiset.ofNatMultiset v h).prod) v.prod
                                                        -/
    ((ofNatMultiset v h).prod : ℕ) = (v.prod : ℕ) := by rw [coe_prod, to_ofNatMultiset]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- If a `Multiset ℕ+` consists only of primes, it can be recast as a `PrimeMultiset`. -/
def ofPNatMultiset (v : Multiset ℕ+) (h : ∀ p : ℕ+, p ∈ v → p.Prime) : PrimeMultiset :=
  @Multiset.pmap ℕ+ Nat.Primes PNat.Prime (fun p hp => ⟨(p : ℕ), hp⟩) v h


theorem to_ofPNatMultiset (v : Multiset ℕ+) (h) : (ofPNatMultiset v h : Multiset ℕ+) = v := by
  /-
    v : Multiset PNat
    h : ∀ (p : PNat), Membership.mem v p → p.Prime
    ⊢ Eq (PrimeMultiset.ofPNatMultiset v h).toPNatMultiset v
  -/
  dsimp [ofPNatMultiset, toPNatMultiset]
  have : (fun (p : ℕ+) (h : p.Prime) => (Coe.coe : Nat.Primes → ℕ+) ⟨p, h⟩) = fun p _ => id p := by
    funext p h
    apply Subtype.eq
    rfl
  /-
    v : Multiset PNat
    h : ∀ (p : PNat), Membership.mem v p → p.Prime
    this : Eq (fun p h => Coe.coe ⟨↑p, h⟩) fun p x => id p
    ⊢ Eq (Multiset.map Coe.coe (Multiset.pmap (fun p hp => ⟨↑p, hp⟩) v h)) v
  -/
  rw [Multiset.map_pmap, this, Multiset.pmap_eq_map, Multiset.map_id]
  /-
    🎉 no goals
  -/


theorem prod_ofPNatMultiset (v : Multiset ℕ+) (h) : ((ofPNatMultiset v h).prod : ℕ+) = v.prod := by
  /-
    v : Multiset PNat
    h : ∀ (p : PNat), Membership.mem v p → p.Prime
    ⊢ Eq (PrimeMultiset.ofPNatMultiset v h).prod v.prod
  -/
  dsimp [prod]
  /-
    v : Multiset PNat
    h : ∀ (p : PNat), Membership.mem v p → p.Prime
    ⊢ Eq (PrimeMultiset.ofPNatMultiset v h).toPNatMultiset.prod v.prod
  -/
  rw [to_ofPNatMultiset]
  /-
    🎉 no goals
  -/


/-- Lists can be coerced to multisets; here we have some results
about how this interacts with our constructions on multisets. -/
def ofNatList (l : List ℕ) (h : ∀ p : ℕ, p ∈ l → p.Prime) : PrimeMultiset :=
  ofNatMultiset (l : Multiset ℕ) h


theorem prod_ofNatList (l : List ℕ) (h) : ((ofNatList l h).prod : ℕ) = l.prod := by
  /-
    l : List Nat
    h : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
    ⊢ Eq (↑(PrimeMultiset.ofNatList l h).prod) l.prod
  -/
  have := prod_ofNatMultiset (l : Multiset ℕ) h
  /-
    l : List Nat
    h : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
    this : Eq (↑(PrimeMultiset.ofNatMultiset (↑l) h).prod) (↑l).prod
    ⊢ Eq (↑(PrimeMultiset.ofNatList l h).prod) l.prod
  -/
  rw [Multiset.prod_coe] at this
  /-
    l : List Nat
    h : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
    this : Eq (↑(PrimeMultiset.ofNatMultiset (↑l) h).prod) l.prod
    ⊢ Eq (↑(PrimeMultiset.ofNatList l h).prod) l.prod
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- If a `List ℕ+` consists only of primes, it can be recast as a `PrimeMultiset` with
the coercion from lists to multisets. -/
def ofPNatList (l : List ℕ+) (h : ∀ p : ℕ+, p ∈ l → p.Prime) : PrimeMultiset :=
  ofPNatMultiset (l : Multiset ℕ+) h


theorem prod_ofPNatList (l : List ℕ+) (h) : (ofPNatList l h).prod = l.prod := by
  /-
    l : List PNat
    h : ∀ (p : PNat), Membership.mem l p → p.Prime
    ⊢ Eq (PrimeMultiset.ofPNatList l h).prod l.prod
  -/
  have := prod_ofPNatMultiset (l : Multiset ℕ+) h
  /-
    l : List PNat
    h : ∀ (p : PNat), Membership.mem l p → p.Prime
    this : Eq (PrimeMultiset.ofPNatMultiset (↑l) h).prod (↑l).prod
    ⊢ Eq (PrimeMultiset.ofPNatList l h).prod l.prod
  -/
  rw [Multiset.prod_coe] at this
  /-
    l : List PNat
    h : ∀ (p : PNat), Membership.mem l p → p.Prime
    this : Eq (PrimeMultiset.ofPNatMultiset (↑l) h).prod l.prod
    ⊢ Eq (PrimeMultiset.ofPNatList l h).prod l.prod
  -/
  exact this
  /-
    🎉 no goals
  -/


/-- The product map gives a homomorphism from the additive monoid
of multisets to the multiplicative monoid ℕ+. -/
theorem prod_zero : (0 : PrimeMultiset).prod = 1 := by
  /-
    ⊢ Eq (PrimeMultiset.prod 0) 1
  -/
  exact Multiset.prod_zero
  /-
    🎉 no goals
  -/


theorem prod_add (u v : PrimeMultiset) : (u + v).prod = u.prod * v.prod := by
  /-
    u v : PrimeMultiset
    ⊢ Eq (HAdd.hAdd u v).prod (HMul.hMul u.prod v.prod)
  -/
  change (coePNatMonoidHom (u + v)).prod = _
  /-
    u v : PrimeMultiset
    ⊢ Eq (PrimeMultiset.coePNatMonoidHom (HAdd.hAdd u v)).prod (HMul.hMul u.prod v …
  -/
  rw [coePNatMonoidHom.map_add]
  /-
    u v : PrimeMultiset
    ⊢ Eq (HAdd.hAdd (PrimeMultiset.coePNatMonoidHom u) (PrimeMultiset.coePNatMonoi …
  -/
  exact Multiset.prod_add _ _
  /-
    🎉 no goals
  -/


theorem prod_smul (d : ℕ) (u : PrimeMultiset) : (d • u).prod = u.prod ^ d := by
  induction d with
  | zero => simp only [zero_nsmul, pow_zero, prod_zero]
  | succ n ih => rw [succ_nsmul, prod_add, ih, pow_succ]


/-- The prime factors of n, regarded as a multiset -/
def factorMultiset (n : ℕ+) : PrimeMultiset :=
  PrimeMultiset.ofNatList (Nat.primeFactorsList n) (@Nat.prime_of_mem_primeFactorsList n)


/-- The product of the factors is the original number -/
theorem prod_factorMultiset (n : ℕ+) : (factorMultiset n).prod = n :=
  eq <| by
    /-
      n : PNat
      ⊢ Eq ↑n.factorMultiset.prod ↑n
    -/
    dsimp [factorMultiset]
    /-
      n : PNat
      ⊢ Eq ↑(PrimeMultiset.ofNatList (↑n).primeFactorsList ⋯).prod ↑n
    -/
    rw [PrimeMultiset.prod_ofNatList]
    /-
      n : PNat
      ⊢ Eq (↑n).primeFactorsList.prod ↑n
    -/
    exact Nat.prod_primeFactorsList n.ne_zero
    /-
      🎉 no goals
    -/


theorem coeNat_factorMultiset (n : ℕ+) :
    (factorMultiset n : Multiset ℕ) = (Nat.primeFactorsList n : Multiset ℕ) :=
  PrimeMultiset.to_ofNatMultiset (Nat.primeFactorsList n) (@Nat.prime_of_mem_primeFactorsList n)


/-- If we start with a multiset of primes, take the product and
 then factor it, we get back the original multiset. -/
theorem factorMultiset_prod (v : PrimeMultiset) : v.prod.factorMultiset = v := by
  /-
    v : PrimeMultiset
    ⊢ Eq v.prod.factorMultiset v
  -/
  apply PrimeMultiset.coeNat_injective
  /-
    case a
    v : PrimeMultiset
    ⊢ Eq (Coe.coe v.prod.factorMultiset) (Coe.coe v)
  -/
  suffices toNatMultiset (PNat.factorMultiset (prod v)) = toNatMultiset v by exact this
  /-
    case a
    v : PrimeMultiset
    ⊢ Eq v.prod.factorMultiset.toNatMultiset v.toNatMultiset
  -/
  rw [v.prod.coeNat_factorMultiset, PrimeMultiset.coe_prod]
  /-
    case a
    v : PrimeMultiset
    ⊢ Eq (↑v.toNatMultiset.prod.primeFactorsList) v.toNatMultiset
  -/
  rcases v with ⟨l⟩
  --unfold_coes
  /-
    case a.mk
    v : PrimeMultiset
    l : List Nat.Primes
    ⊢ Eq (↑(PrimeMultiset.toNatMultiset (Quot.mk (⇑(List.isSetoid Nat.Primes)) l)) …
  -/
  dsimp [PrimeMultiset.toNatMultiset]
  /-
    case a.mk
    v : PrimeMultiset
    l : List Nat.Primes
    ⊢ Eq ↑(List.map Coe.coe l).prod.primeFactorsList ↑(List.map Coe.coe l)
  -/
  let l' := l.map (Coe.coe : Nat.Primes → ℕ)
  have : ∀ p : ℕ, p ∈ l' → p.Prime := fun p hp => by
    rcases List.mem_map.mp hp with ⟨⟨_, hp'⟩, ⟨_, h_eq⟩⟩
    exact h_eq ▸ hp'
  /-
    case a.mk
    v : PrimeMultiset
    l : List Nat.Primes
    l' : List Nat := List.map Coe.coe l
    this : ∀ (p : Nat), Membership.mem l' p → Nat.Prime p
    ⊢ Eq ↑(List.map Coe.coe l).prod.primeFactorsList ↑(List.map Coe.coe l)
  -/
  exact Multiset.coe_eq_coe.mpr (@Nat.primeFactorsList_unique _ l' rfl this).symm
  /-
    🎉 no goals
  -/


/-- Positive integers biject with multisets of primes. -/
def factorMultisetEquiv : ℕ+ ≃ PrimeMultiset where
  toFun := factorMultiset
  invFun := PrimeMultiset.prod
  left_inv := prod_factorMultiset
  right_inv := PrimeMultiset.factorMultiset_prod


/-- Factoring gives a homomorphism from the multiplicative
 monoid ℕ+ to the additive monoid of multisets. -/
theorem factorMultiset_one : factorMultiset 1 = 0 := by
  /-
    ⊢ Eq (PNat.factorMultiset 1) 0
  -/
  simp [factorMultiset, PrimeMultiset.ofNatList, PrimeMultiset.ofNatMultiset]
  /-
    🎉 no goals
  -/


theorem factorMultiset_mul (n m : ℕ+) :
    factorMultiset (n * m) = factorMultiset n + factorMultiset m := by
  /-
    n m : PNat
    ⊢ Eq (HMul.hMul n m).factorMultiset (HAdd.hAdd n.factorMultiset m.factorMultis …
  -/
  let u := factorMultiset n
  /-
    n m : PNat
    u : PrimeMultiset := n.factorMultiset
    ⊢ Eq (HMul.hMul n m).factorMultiset (HAdd.hAdd n.factorMultiset m.factorMultis …
  -/
  let v := factorMultiset m
  /-
    n m : PNat
    u : PrimeMultiset := n.factorMultiset
    v : PrimeMultiset := m.factorMultiset
    ⊢ Eq (HMul.hMul n m).factorMultiset (HAdd.hAdd n.factorMultiset m.factorMultis …
  -/
  have : n = u.prod := (prod_factorMultiset n).symm; rw [this]
  /-
    n m : PNat
    u : PrimeMultiset := n.factorMultiset
    v : PrimeMultiset := m.factorMultiset
    this : Eq n u.prod
    ⊢ Eq (HMul.hMul u.prod m).factorMultiset (HAdd.hAdd u.prod.factorMultiset m.fa …
  -/
  have : m = v.prod := (prod_factorMultiset m).symm; rw [this]
  /-
    n m : PNat
    u : PrimeMultiset := n.factorMultiset
    v : PrimeMultiset := m.factorMultiset
    this✝ : Eq n u.prod
    this : Eq m v.prod
    ⊢ Eq (HMul.hMul u.prod v.prod).factorMultiset (HAdd.hAdd u.prod.factorMultiset …
  -/
  rw [← PrimeMultiset.prod_add]
  /-
    n m : PNat
    u : PrimeMultiset := n.factorMultiset
    v : PrimeMultiset := m.factorMultiset
    this✝ : Eq n u.prod
    this : Eq m v.prod
    ⊢ Eq (HAdd.hAdd u v).prod.factorMultiset (HAdd.hAdd u.prod.factorMultiset v.pr …
  -/
  repeat' rw [PrimeMultiset.factorMultiset_prod]
  /-
    🎉 no goals
  -/


theorem factorMultiset_pow (n : ℕ+) (m : ℕ) :
    factorMultiset (n ^ m) = m • factorMultiset n := by
  /-
    n : PNat
    m : Nat
    ⊢ Eq (HPow.hPow n m).factorMultiset (HSMul.hSMul m n.factorMultiset)
  -/
  let u := factorMultiset n
  /-
    n : PNat
    m : Nat
    u : PrimeMultiset := n.factorMultiset
    ⊢ Eq (HPow.hPow n m).factorMultiset (HSMul.hSMul m n.factorMultiset)
  -/
  have : n = u.prod := (prod_factorMultiset n).symm
  /-
    n : PNat
    m : Nat
    u : PrimeMultiset := n.factorMultiset
    this : Eq n u.prod
    ⊢ Eq (HPow.hPow n m).factorMultiset (HSMul.hSMul m n.factorMultiset)
  -/
  rw [this, ← PrimeMultiset.prod_smul]
  /-
    n : PNat
    m : Nat
    u : PrimeMultiset := n.factorMultiset
    this : Eq n u.prod
    ⊢ Eq (HSMul.hSMul m u).prod.factorMultiset (HSMul.hSMul m u.prod.factorMultiset)
  -/
  repeat' rw [PrimeMultiset.factorMultiset_prod]
  /-
    🎉 no goals
  -/


/-- Factoring a prime gives the corresponding one-element multiset. -/
theorem factorMultiset_ofPrime (p : Nat.Primes) :
    (p : ℕ+).factorMultiset = PrimeMultiset.ofPrime p := by
  /-
    p : Nat.Primes
    ⊢ Eq (↑p).factorMultiset (PrimeMultiset.ofPrime p)
  -/
  apply factorMultisetEquiv.symm.injective
  /-
    case a
    p : Nat.Primes
    ⊢ Eq (PNat.factorMultisetEquiv.symm (↑p).factorMultiset) (PNat.factorMultisetE …
  -/
  change (p : ℕ+).factorMultiset.prod = (PrimeMultiset.ofPrime p).prod
  /-
    case a
    p : Nat.Primes
    ⊢ Eq (↑p).factorMultiset.prod (PrimeMultiset.ofPrime p).prod
  -/
  rw [(p : ℕ+).prod_factorMultiset, PrimeMultiset.prod_ofPrime]
  /-
    🎉 no goals
  -/


/-- We now have four different results that all encode the
 idea that inequality of multisets corresponds to divisibility
 of positive integers. -/
theorem factorMultiset_le_iff {m n : ℕ+} : factorMultiset m ≤ factorMultiset n ↔ m ∣ n := by
  /-
    m n : PNat
    ⊢ Iff (LE.le m.factorMultiset n.factorMultiset) (Dvd.dvd m n)
  -/
  constructor
    /-
      case mp
      m n : PNat
      ⊢ LE.le m.factorMultiset n.factorMultiset → Dvd.dvd m n
    -/
  · intro h
    /-
      case mp
      m n : PNat
      h : LE.le m.factorMultiset n.factorMultiset
      ⊢ Dvd.dvd m n
    -/
    rw [← prod_factorMultiset m, ← prod_factorMultiset m]
    /-
      case mp
      m n : PNat
      h : LE.le m.factorMultiset n.factorMultiset
      ⊢ Dvd.dvd m.factorMultiset.prod.factorMultiset.prod n
    -/
    apply Dvd.intro (n.factorMultiset - m.factorMultiset).prod
    rw [← PrimeMultiset.prod_add, PrimeMultiset.factorMultiset_prod, add_tsub_cancel_of_le h,
      prod_factorMultiset]
    /-
      case mpr
      m n : PNat
      ⊢ Dvd.dvd m n → LE.le m.factorMultiset n.factorMultiset
    -/
  · intro h
    /-
      case mpr
      m n : PNat
      h : Dvd.dvd m n
      ⊢ LE.le m.factorMultiset n.factorMultiset
    -/
    rw [← mul_div_exact h, factorMultiset_mul]
    /-
      case mpr
      m n : PNat
      h : Dvd.dvd m n
      ⊢ LE.le m.factorMultiset (HAdd.hAdd m.factorMultiset (n.divExact m).factorMult …
    -/
    exact le_self_add
    /-
      🎉 no goals
    -/


theorem factorMultiset_le_iff' {m : ℕ+} {v : PrimeMultiset} :
    factorMultiset m ≤ v ↔ m ∣ v.prod := by
  /-
    m : PNat
    v : PrimeMultiset
    ⊢ Iff (LE.le m.factorMultiset v) (Dvd.dvd m v.prod)
  -/
  let h := @factorMultiset_le_iff m v.prod
  /-
    m : PNat
    v : PrimeMultiset
    h : Iff (LE.le m.factorMultiset v.prod.factorMultiset) (Dvd.dvd m v.prod) := P …
    ⊢ Iff (LE.le m.factorMultiset v) (Dvd.dvd m v.prod)
  -/
  rw [v.factorMultiset_prod] at h
  /-
    m : PNat
    v : PrimeMultiset
    h : Iff (LE.le m.factorMultiset v) (Dvd.dvd m v.prod)
    ⊢ Iff (LE.le m.factorMultiset v) (Dvd.dvd m v.prod)
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem prod_dvd_iff {u v : PrimeMultiset} : u.prod ∣ v.prod ↔ u ≤ v := by
  /-
    u v : PrimeMultiset
    ⊢ Iff (Dvd.dvd u.prod v.prod) (LE.le u v)
  -/
  let h := @PNat.factorMultiset_le_iff' u.prod v
  /-
    u v : PrimeMultiset
    h : Iff (LE.le u.prod.factorMultiset v) (Dvd.dvd u.prod v.prod) := PNat.factor …
    ⊢ Iff (Dvd.dvd u.prod v.prod) (LE.le u v)
  -/
  rw [u.factorMultiset_prod] at h
  /-
    u v : PrimeMultiset
    h : Iff (LE.le u v) (Dvd.dvd u.prod v.prod)
    ⊢ Iff (Dvd.dvd u.prod v.prod) (LE.le u v)
  -/
  exact h.symm
  /-
    🎉 no goals
  -/


theorem prod_dvd_iff' {u : PrimeMultiset} {n : ℕ+} : u.prod ∣ n ↔ u ≤ n.factorMultiset := by
  /-
    u : PrimeMultiset
    n : PNat
    ⊢ Iff (Dvd.dvd u.prod n) (LE.le u n.factorMultiset)
  -/
  let h := @prod_dvd_iff u n.factorMultiset
  /-
    u : PrimeMultiset
    n : PNat
    h : Iff (Dvd.dvd u.prod n.factorMultiset.prod) (LE.le u n.factorMultiset) := P …
    ⊢ Iff (Dvd.dvd u.prod n) (LE.le u n.factorMultiset)
  -/
  rw [n.prod_factorMultiset] at h
  /-
    u : PrimeMultiset
    n : PNat
    h : Iff (Dvd.dvd u.prod n) (LE.le u n.factorMultiset)
    ⊢ Iff (Dvd.dvd u.prod n) (LE.le u n.factorMultiset)
  -/
  exact h
  /-
    🎉 no goals
  -/


/-- The gcd and lcm operations on positive integers correspond
 to the inf and sup operations on multisets. -/
theorem factorMultiset_gcd (m n : ℕ+) :
    factorMultiset (gcd m n) = factorMultiset m ⊓ factorMultiset n := by
  /-
    m n : PNat
    ⊢ Eq (m.gcd n).factorMultiset (Min.min m.factorMultiset n.factorMultiset)
  -/
  apply le_antisymm
    /-
      case a
      m n : PNat
      ⊢ LE.le (m.gcd n).factorMultiset (Min.min m.factorMultiset n.factorMultiset)
    -/
  · apply le_inf_iff.mpr; constructor <;> apply factorMultiset_le_iff.mpr
      /-
        case a.left
        m n : PNat
        ⊢ Dvd.dvd (m.gcd n) m
      -/
    · exact gcd_dvd_left m n
      /-
        🎉 no goals
      -/
      /-
        case a.right
        m n : PNat
        ⊢ Dvd.dvd (m.gcd n) n
      -/
    · exact gcd_dvd_right m n
      /-
        🎉 no goals
      -/
    /-
      case a
      m n : PNat
      ⊢ LE.le (Min.min m.factorMultiset n.factorMultiset) (m.gcd n).factorMultiset
    -/
  · rw [← PrimeMultiset.prod_dvd_iff, prod_factorMultiset]
    /-
      case a
      m n : PNat
      ⊢ Dvd.dvd (Min.min m.factorMultiset n.factorMultiset).prod (m.gcd n)
    -/
    apply dvd_gcd <;> rw [PrimeMultiset.prod_dvd_iff']
      /-
        case a.hm
        m n : PNat
        ⊢ LE.le (Min.min m.factorMultiset n.factorMultiset) m.factorMultiset
      -/
    · exact inf_le_left
      /-
        🎉 no goals
      -/
      /-
        case a.hn
        m n : PNat
        ⊢ LE.le (Min.min m.factorMultiset n.factorMultiset) n.factorMultiset
      -/
    · exact inf_le_right
      /-
        🎉 no goals
      -/


theorem factorMultiset_lcm (m n : ℕ+) :
    factorMultiset (lcm m n) = factorMultiset m ⊔ factorMultiset n := by
  /-
    m n : PNat
    ⊢ Eq (m.lcm n).factorMultiset (Max.max m.factorMultiset n.factorMultiset)
  -/
  apply le_antisymm
    /-
      case a
      m n : PNat
      ⊢ LE.le (m.lcm n).factorMultiset (Max.max m.factorMultiset n.factorMultiset)
    -/
  · rw [← PrimeMultiset.prod_dvd_iff, prod_factorMultiset]
    /-
      case a
      m n : PNat
      ⊢ Dvd.dvd (m.lcm n) (Max.max m.factorMultiset n.factorMultiset).prod
    -/
    apply lcm_dvd <;> rw [← factorMultiset_le_iff']
      /-
        case a.hm
        m n : PNat
        ⊢ LE.le m.factorMultiset (Max.max m.factorMultiset n.factorMultiset)
      -/
    · exact le_sup_left
      /-
        🎉 no goals
      -/
      /-
        case a.hn
        m n : PNat
        ⊢ LE.le n.factorMultiset (Max.max m.factorMultiset n.factorMultiset)
      -/
    · exact le_sup_right
      /-
        🎉 no goals
      -/
    /-
      case a
      m n : PNat
      ⊢ LE.le (Max.max m.factorMultiset n.factorMultiset) (m.lcm n).factorMultiset
    -/
  · apply sup_le_iff.mpr; constructor <;> apply factorMultiset_le_iff.mpr
      /-
        case a.left
        m n : PNat
        ⊢ Dvd.dvd m (m.lcm n)
      -/
    · exact dvd_lcm_left m n
      /-
        🎉 no goals
      -/
      /-
        case a.right
        m n : PNat
        ⊢ Dvd.dvd n (m.lcm n)
      -/
    · exact dvd_lcm_right m n
      /-
        🎉 no goals
      -/


/-- The number of occurrences of p in the factor multiset of m
 is the same as the p-adic valuation of m. -/
theorem count_factorMultiset (m : ℕ+) (p : Nat.Primes) (k : ℕ) :
    (p : ℕ+) ^ k ∣ m ↔ k ≤ m.factorMultiset.count p := by
  rw [Multiset.le_count_iff_replicate_le, ← factorMultiset_le_iff, factorMultiset_pow,
    factorMultiset_ofPrime]
  /-
    m : PNat
    p : Nat.Primes
    k : Nat
    ⊢ Iff (LE.le (HSMul.hSMul k (PrimeMultiset.ofPrime p)) m.factorMultiset) (LE.l …
  -/
  congr! 2
  /-
    case a.h.e'_3.h
    m : PNat
    p : Nat.Primes
    k : Nat
    e_1✝ : Eq PrimeMultiset (Multiset Nat.Primes)
    ⊢ Eq (HSMul.hSMul k (PrimeMultiset.ofPrime p)) (Multiset.replicate k p)
  -/
  apply Multiset.eq_replicate.mpr
  /-
    case a.h.e'_3.h
    m : PNat
    p : Nat.Primes
    k : Nat
    e_1✝ : Eq PrimeMultiset (Multiset Nat.Primes)
    ⊢ And (Eq (Multiset.card (HSMul.hSMul k (PrimeMultiset.ofPrime p))) k) (∀ (b : …
  -/
  constructor
    /-
      case a.h.e'_3.h.left
      m : PNat
      p : Nat.Primes
      k : Nat
      e_1✝ : Eq PrimeMultiset (Multiset Nat.Primes)
      ⊢ Eq (Multiset.card (HSMul.hSMul k (PrimeMultiset.ofPrime p))) k
    -/
  · rw [Multiset.card_nsmul, PrimeMultiset.card_ofPrime, mul_one]
    /-
      🎉 no goals
    -/
    /-
      case a.h.e'_3.h.right
      m : PNat
      p : Nat.Primes
      k : Nat
      e_1✝ : Eq PrimeMultiset (Multiset Nat.Primes)
      ⊢ ∀ (b : Nat.Primes), Membership.mem (HSMul.hSMul k (PrimeMultiset.ofPrime p)) …
    -/
  · intro q h
    /-
      case a.h.e'_3.h.right
      m : PNat
      p : Nat.Primes
      k : Nat
      e_1✝ : Eq PrimeMultiset (Multiset Nat.Primes)
      q : Nat.Primes
      h : Membership.mem (HSMul.hSMul k (PrimeMultiset.ofPrime p)) q
      ⊢ Eq q p
    -/
    rw [PrimeMultiset.ofPrime, Multiset.nsmul_singleton _ k] at h
    /-
      case a.h.e'_3.h.right
      m : PNat
      p : Nat.Primes
      k : Nat
      e_1✝ : Eq PrimeMultiset (Multiset Nat.Primes)
      q : Nat.Primes
      h : Membership.mem (Multiset.replicate k p) q
      ⊢ Eq q p
    -/
    exact Multiset.eq_of_mem_replicate h
    /-
      🎉 no goals
    -/


theorem prod_inf (u v : PrimeMultiset) : (u ⊓ v).prod = PNat.gcd u.prod v.prod := by
  /-
    u v : PrimeMultiset
    ⊢ Eq (Min.min u v).prod (u.prod.gcd v.prod)
  -/
  let n := u.prod
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    ⊢ Eq (Min.min u v).prod (u.prod.gcd v.prod)
  -/
  let m := v.prod
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    ⊢ Eq (Min.min u v).prod (u.prod.gcd v.prod)
  -/
  change (u ⊓ v).prod = PNat.gcd n m
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    ⊢ Eq (Min.min u v).prod (n.gcd m)
  -/
  have : u = n.factorMultiset := u.factorMultiset_prod.symm; rw [this]
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    this : Eq u n.factorMultiset
    ⊢ Eq (Min.min n.factorMultiset v).prod (n.gcd m)
  -/
  have : v = m.factorMultiset := v.factorMultiset_prod.symm; rw [this]
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    this✝ : Eq u n.factorMultiset
    this : Eq v m.factorMultiset
    ⊢ Eq (Min.min n.factorMultiset m.factorMultiset).prod (n.gcd m)
  -/
  rw [← PNat.factorMultiset_gcd n m, PNat.prod_factorMultiset]
  /-
    🎉 no goals
  -/


theorem prod_sup (u v : PrimeMultiset) : (u ⊔ v).prod = PNat.lcm u.prod v.prod := by
  /-
    u v : PrimeMultiset
    ⊢ Eq (Max.max u v).prod (u.prod.lcm v.prod)
  -/
  let n := u.prod
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    ⊢ Eq (Max.max u v).prod (u.prod.lcm v.prod)
  -/
  let m := v.prod
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    ⊢ Eq (Max.max u v).prod (u.prod.lcm v.prod)
  -/
  change (u ⊔ v).prod = PNat.lcm n m
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    ⊢ Eq (Max.max u v).prod (n.lcm m)
  -/
  have : u = n.factorMultiset := u.factorMultiset_prod.symm; rw [this]
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    this : Eq u n.factorMultiset
    ⊢ Eq (Max.max n.factorMultiset v).prod (n.lcm m)
  -/
  have : v = m.factorMultiset := v.factorMultiset_prod.symm; rw [this]
  /-
    u v : PrimeMultiset
    n : PNat := u.prod
    m : PNat := v.prod
    this✝ : Eq u n.factorMultiset
    this : Eq v m.factorMultiset
    ⊢ Eq (Max.max n.factorMultiset m.factorMultiset).prod (n.lcm m)
  -/
  rw [← PNat.factorMultiset_lcm n m, PNat.prod_factorMultiset]
  /-
    🎉 no goals
  -/


