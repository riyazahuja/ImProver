/-- `IsSubfield (S : Set F)` is the predicate saying that a given subset of a field is
the set underlying a subfield. This structure is deprecated; use the bundled variant
`Subfield F` to model subfields of a field. -/
structure IsSubfield extends IsSubring S : Prop where
  inv_mem : ∀ {x : F}, x ∈ S → x⁻¹ ∈ S


theorem IsSubfield.div_mem {S : Set F} (hS : IsSubfield S) {x y : F} (hx : x ∈ S) (hy : y ∈ S) :
    x / y ∈ S := by
  /-
    F : Type u_1
    inst✝ : Field F
    S : Set F
    hS : IsSubfield S
    x y : F
    hx : Membership.mem S x
    hy : Membership.mem S y
    ⊢ Membership.mem S (HDiv.hDiv x y)
  -/
  rw [div_eq_mul_inv]
  /-
    F : Type u_1
    inst✝ : Field F
    S : Set F
    hS : IsSubfield S
    x y : F
    hx : Membership.mem S x
    hy : Membership.mem S y
    ⊢ Membership.mem S (HMul.hMul x (Inv.inv y))
  -/
  exact hS.toIsSubring.toIsSubmonoid.mul_mem hx (hS.inv_mem hy)
  /-
    🎉 no goals
  -/


theorem IsSubfield.pow_mem {a : F} {n : ℤ} {s : Set F} (hs : IsSubfield s) (h : a ∈ s) :
    a ^ n ∈ s := by
  /-
    F : Type u_1
    inst✝ : Field F
    a : F
    n : Int
    s : Set F
    hs : IsSubfield s
    h : Membership.mem s a
    ⊢ Membership.mem s (HPow.hPow a n)
  -/
  cases' n with n n
    /-
      case ofNat
      F : Type u_1
      inst✝ : Field F
      a : F
      s : Set F
      hs : IsSubfield s
      h : Membership.mem s a
      n : Nat
      ⊢ Membership.mem s (HPow.hPow a (Int.ofNat n))
    -/
  · suffices a ^ (n : ℤ) ∈ s by exact this
    /-
      case ofNat
      F : Type u_1
      inst✝ : Field F
      a : F
      s : Set F
      hs : IsSubfield s
      h : Membership.mem s a
      n : Nat
      ⊢ Membership.mem s (HPow.hPow a ↑n)
    -/
    rw [zpow_natCast]
    /-
      case ofNat
      F : Type u_1
      inst✝ : Field F
      a : F
      s : Set F
      hs : IsSubfield s
      h : Membership.mem s a
      n : Nat
      ⊢ Membership.mem s (HPow.hPow a n)
    -/
    exact hs.toIsSubring.toIsSubmonoid.pow_mem h
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      F : Type u_1
      inst✝ : Field F
      a : F
      s : Set F
      hs : IsSubfield s
      h : Membership.mem s a
      n : Nat
      ⊢ Membership.mem s (HPow.hPow a (Int.negSucc n))
    -/
  · rw [zpow_negSucc]
    /-
      case negSucc
      F : Type u_1
      inst✝ : Field F
      a : F
      s : Set F
      hs : IsSubfield s
      h : Membership.mem s a
      n : Nat
      ⊢ Membership.mem s (Inv.inv (HPow.hPow a (HAdd.hAdd n 1)))
    -/
    exact hs.inv_mem (hs.toIsSubring.toIsSubmonoid.pow_mem h)
    /-
      🎉 no goals
    -/


theorem Univ.isSubfield : IsSubfield (@Set.univ F) :=
  { Univ.isSubmonoid, IsAddSubgroup.univ_addSubgroup with
    inv_mem := fun _ ↦ trivial }


theorem Preimage.isSubfield {K : Type*} [Field K] (f : F →+* K) {s : Set K} (hs : IsSubfield s) :
    IsSubfield (f ⁻¹' s) :=
  { f.isSubring_preimage hs.toIsSubring with
    inv_mem := fun {a} (ha : f a ∈ s) ↦ show f a⁻¹ ∈ s by
      /-
        F : Type u_1
        inst✝¹ : Field F
        K : Type u_2
        inst✝ : Field K
        f : RingHom F K
        s : Set K
        hs : IsSubfield s
        a : F
        ha : Membership.mem s (f a)
        ⊢ Membership.mem s (f (Inv.inv a))
      -/
      rw [map_inv₀]
      /-
        F : Type u_1
        inst✝¹ : Field F
        K : Type u_2
        inst✝ : Field K
        f : RingHom F K
        s : Set K
        hs : IsSubfield s
        a : F
        ha : Membership.mem s (f a)
        ⊢ Membership.mem s (Inv.inv (f a))
      -/
      exact hs.inv_mem ha }
      /-
        🎉 no goals
      -/


theorem Image.isSubfield {K : Type*} [Field K] (f : F →+* K) {s : Set F} (hs : IsSubfield s) :
    IsSubfield (f '' s) :=
  { f.isSubring_image hs.toIsSubring with
    inv_mem := fun ⟨x, xmem, ha⟩ ↦ ⟨x⁻¹, hs.inv_mem xmem, ha ▸ map_inv₀ f x⟩ }


theorem Range.isSubfield {K : Type*} [Field K] (f : F →+* K) : IsSubfield (Set.range f) := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    K : Type u_2
    inst✝ : Field K
    f : RingHom F K
    ⊢ IsSubfield (Set.range ⇑f)
  -/
  rw [← Set.image_univ]
  /-
    F : Type u_1
    inst✝¹ : Field F
    K : Type u_2
    inst✝ : Field K
    f : RingHom F K
    ⊢ IsSubfield (Set.image (⇑f) Set.univ)
  -/
  apply Image.isSubfield _ Univ.isSubfield
  /-
    🎉 no goals
  -/


/-- `Field.closure s` is the minimal subfield that includes `s`. -/
def closure : Set F :=
  { x | ∃ y ∈ Ring.closure S, ∃ z ∈ Ring.closure S, y / z = x }


theorem ring_closure_subset : Ring.closure S ⊆ closure S :=
  fun x hx ↦ ⟨x, hx, 1, Ring.closure.isSubring.toIsSubmonoid.one_mem, div_one x⟩


theorem closure.isSubmonoid : IsSubmonoid (closure S) :=
  { mul_mem := by
      /-
        F : Type u_1
        inst✝ : Field F
        S : Set F
        ⊢ ∀ {a b : F}, Membership.mem (Field.closure S) a → Membership.mem (Field.clos …
      -/
      rintro _ _ ⟨p, hp, q, hq, hq0, rfl⟩ ⟨r, hr, s, hs, hs0, rfl⟩
      exact ⟨p * r, IsSubmonoid.mul_mem Ring.closure.isSubring.toIsSubmonoid hp hr, q * s,
        IsSubmonoid.mul_mem Ring.closure.isSubring.toIsSubmonoid hq hs,
        (div_mul_div_comm _ _ _ _).symm⟩
    one_mem := ring_closure_subset <| IsSubmonoid.one_mem Ring.closure.isSubring.toIsSubmonoid }


theorem closure.isSubfield : IsSubfield (closure S) :=
  { closure.isSubmonoid with
    add_mem := by
      /-
        F : Type u_1
        inst✝ : Field F
        S : Set F
        ⊢ ∀ {a b : F}, Membership.mem (Field.closure S) a → Membership.mem (Field.clos …
      -/
      intro a b ha hb
      /-
        F : Type u_1
        inst✝ : Field F
        S : Set F
        a b : F
        ha : Membership.mem (Field.closure S) a
        hb : Membership.mem (Field.closure S) b
        ⊢ Membership.mem (Field.closure S) (HAdd.hAdd a b)
      -/
      rcases id ha with ⟨p, hp, q, hq, rfl⟩
      /-
        case intro.intro.intro.intro
        F : Type u_1
        inst✝ : Field F
        S : Set F
        b : F
        hb : Membership.mem (Field.closure S) b
        p : F
        hp : Membership.mem (Ring.closure S) p
        q : F
        hq : Membership.mem (Ring.closure S) q
        ha : Membership.mem (Field.closure S) (HDiv.hDiv p q)
        ⊢ Membership.mem (Field.closure S) (HAdd.hAdd (HDiv.hDiv p q) b)
      -/
      rcases id hb with ⟨r, hr, s, hs, rfl⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        F : Type u_1
        inst✝ : Field F
        S : Set F
        p : F
        hp : Membership.mem (Ring.closure S) p
        q : F
        hq : Membership.mem (Ring.closure S) q
        ha : Membership.mem (Field.closure S) (HDiv.hDiv p q)
        r : F
        hr : Membership.mem (Ring.closure S) r
        s : F
        hs : Membership.mem (Ring.closure S) s
        hb : Membership.mem (Field.closure S) (HDiv.hDiv r s)
        ⊢ Membership.mem (Field.closure S) (HAdd.hAdd (HDiv.hDiv p q) (HDiv.hDiv r s))
      -/
      by_cases hq0 : q = 0
        /-
          case pos
          F : Type u_1
          inst✝ : Field F
          S : Set F
          p : F
          hp : Membership.mem (Ring.closure S) p
          q : F
          hq : Membership.mem (Ring.closure S) q
          ha : Membership.mem (Field.closure S) (HDiv.hDiv p q)
          r : F
          hr : Membership.mem (Ring.closure S) r
          s : F
          hs : Membership.mem (Ring.closure S) s
          hb : Membership.mem (Field.closure S) (HDiv.hDiv r s)
          hq0 : Eq q 0
          ⊢ Membership.mem (Field.closure S) (HAdd.hAdd (HDiv.hDiv p q) (HDiv.hDiv r s))
        -/
      · rwa [hq0, div_zero, zero_add]
        /-
          🎉 no goals
        -/
      /-
        case neg
        F : Type u_1
        inst✝ : Field F
        S : Set F
        p : F
        hp : Membership.mem (Ring.closure S) p
        q : F
        hq : Membership.mem (Ring.closure S) q
        ha : Membership.mem (Field.closure S) (HDiv.hDiv p q)
        r : F
        hr : Membership.mem (Ring.closure S) r
        s : F
        hs : Membership.mem (Ring.closure S) s
        hb : Membership.mem (Field.closure S) (HDiv.hDiv r s)
        hq0 : Not (Eq q 0)
        ⊢ Membership.mem (Field.closure S) (HAdd.hAdd (HDiv.hDiv p q) (HDiv.hDiv r s))
      -/
      by_cases hs0 : s = 0
        /-
          case pos
          F : Type u_1
          inst✝ : Field F
          S : Set F
          p : F
          hp : Membership.mem (Ring.closure S) p
          q : F
          hq : Membership.mem (Ring.closure S) q
          ha : Membership.mem (Field.closure S) (HDiv.hDiv p q)
          r : F
          hr : Membership.mem (Ring.closure S) r
          s : F
          hs : Membership.mem (Ring.closure S) s
          hb : Membership.mem (Field.closure S) (HDiv.hDiv r s)
          hq0 : Not (Eq q 0)
          hs0 : Eq s 0
          ⊢ Membership.mem (Field.closure S) (HAdd.hAdd (HDiv.hDiv p q) (HDiv.hDiv r s))
        -/
      · rwa [hs0, div_zero, add_zero]
        /-
          🎉 no goals
        -/
      exact ⟨p * s + q * r,
        IsAddSubmonoid.add_mem Ring.closure.isSubring.toIsAddSubgroup.toIsAddSubmonoid
          (Ring.closure.isSubring.toIsSubmonoid.mul_mem hp hs)
          (Ring.closure.isSubring.toIsSubmonoid.mul_mem hq hr),
        q * s, Ring.closure.isSubring.toIsSubmonoid.mul_mem hq hs, (div_add_div p r hq0 hs0).symm⟩
    zero_mem := ring_closure_subset Ring.closure.isSubring.toIsAddSubgroup.toIsAddSubmonoid.zero_mem
    neg_mem := by
      /-
        F : Type u_1
        inst✝ : Field F
        S : Set F
        ⊢ ∀ {a : F}, Membership.mem (Field.closure S) a → Membership.mem (Field.closur …
      -/
      rintro _ ⟨p, hp, q, hq, rfl⟩
      /-
        case intro.intro.intro.intro
        F : Type u_1
        inst✝ : Field F
        S : Set F
        p : F
        hp : Membership.mem (Ring.closure S) p
        q : F
        hq : Membership.mem (Ring.closure S) q
        ⊢ Membership.mem (Field.closure S) (Neg.neg (HDiv.hDiv p q))
      -/
      exact ⟨-p, Ring.closure.isSubring.toIsAddSubgroup.neg_mem hp, q, hq, neg_div q p⟩
      /-
        🎉 no goals
      -/
    inv_mem := by
      /-
        F : Type u_1
        inst✝ : Field F
        S : Set F
        ⊢ ∀ {x : F}, Membership.mem (Field.closure S) x → Membership.mem (Field.closur …
      -/
      rintro _ ⟨p, hp, q, hq, rfl⟩
      /-
        case intro.intro.intro.intro
        F : Type u_1
        inst✝ : Field F
        S : Set F
        p : F
        hp : Membership.mem (Ring.closure S) p
        q : F
        hq : Membership.mem (Ring.closure S) q
        ⊢ Membership.mem (Field.closure S) (Inv.inv (HDiv.hDiv p q))
      -/
      exact ⟨q, hq, p, hp, (inv_div _ _).symm⟩ }
      /-
        🎉 no goals
      -/


theorem mem_closure {a : F} (ha : a ∈ S) : a ∈ closure S :=
  ring_closure_subset <| Ring.mem_closure ha


theorem subset_closure : S ⊆ closure S :=
  fun _ ↦ mem_closure


theorem closure_subset {T : Set F} (hT : IsSubfield T) (H : S ⊆ T) : closure S ⊆ T := by
  /-
    F : Type u_1
    inst✝ : Field F
    S T : Set F
    hT : IsSubfield T
    H : HasSubset.Subset S T
    ⊢ HasSubset.Subset (Field.closure S) T
  -/
  rintro _ ⟨p, hp, q, hq, hq0, rfl⟩
  exact hT.div_mem (Ring.closure_subset hT.toIsSubring H hp)
    (Ring.closure_subset hT.toIsSubring H hq)


theorem closure_subset_iff {s t : Set F} (ht : IsSubfield t) : closure s ⊆ t ↔ s ⊆ t :=
  ⟨Set.Subset.trans subset_closure, closure_subset ht⟩


@[gcongr]
theorem closure_mono {s t : Set F} (H : s ⊆ t) : closure s ⊆ closure t :=
  closure_subset closure.isSubfield <| Set.Subset.trans H subset_closure


theorem isSubfield_iUnion_of_directed {ι : Type*} [Nonempty ι] {s : ι → Set F}
    (hs : ∀ i, IsSubfield (s i)) (directed : ∀ i j, ∃ k, s i ⊆ s k ∧ s j ⊆ s k) :
    IsSubfield (⋃ i, s i) :=
  { inv_mem := fun hx ↦
      let ⟨i, hi⟩ := Set.mem_iUnion.1 hx
      Set.mem_iUnion.2 ⟨i, (hs i).inv_mem hi⟩
    toIsSubring := isSubring_iUnion_of_directed (fun i ↦ (hs i).toIsSubring) directed }


theorem IsSubfield.inter {S₁ S₂ : Set F} (hS₁ : IsSubfield S₁) (hS₂ : IsSubfield S₂) :
    IsSubfield (S₁ ∩ S₂) :=
  { IsSubring.inter hS₁.toIsSubring hS₂.toIsSubring with
    inv_mem := fun hx ↦ ⟨hS₁.inv_mem hx.1, hS₂.inv_mem hx.2⟩ }


theorem IsSubfield.iInter {ι : Sort*} {S : ι → Set F} (h : ∀ y : ι, IsSubfield (S y)) :
    IsSubfield (Set.iInter S) :=
  { IsSubring.iInter fun y ↦ (h y).toIsSubring with
    inv_mem := fun hx ↦ Set.mem_iInter.2 fun y ↦ (h y).inv_mem <| Set.mem_iInter.1 hx y }

