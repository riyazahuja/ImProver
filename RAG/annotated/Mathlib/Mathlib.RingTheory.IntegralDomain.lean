theorem mul_right_bijective_of_finite₀ {a : M} (ha : a ≠ 0) : Bijective fun b => a * b :=
  Finite.injective_iff_bijective.1 <| mul_right_injective₀ ha


theorem mul_left_bijective_of_finite₀ {a : M} (ha : a ≠ 0) : Bijective fun b => b * a :=
  Finite.injective_iff_bijective.1 <| mul_left_injective₀ ha


/-- Every finite nontrivial cancel_monoid_with_zero is a group_with_zero. -/
def Fintype.groupWithZeroOfCancel (M : Type*) [CancelMonoidWithZero M] [DecidableEq M] [Fintype M]
    [Nontrivial M] : GroupWithZero M :=
  { ‹Nontrivial M›,
    ‹CancelMonoidWithZero M› with
    inv := fun a => if h : a = 0 then 0 else Fintype.bijInv (mul_right_bijective_of_finite₀ h) 1
    mul_inv_cancel := fun a ha => by
      /-
        M✝ : Type u_1
        inst✝⁵ : CancelMonoidWithZero M✝
        inst✝⁴ : Finite M✝
        M : Type u_2
        inst✝³ : CancelMonoidWithZero M
        inst✝² : DecidableEq M
        inst✝¹ : Fintype M
        inst✝ : Nontrivial M
        a : M
        ha : Ne a 0
        ⊢ Eq (HMul.hMul a (Inv.inv a)) 1
      -/
      simp only [Inv.inv, dif_neg ha]
                   /-
                     M✝ : Type u_1
                     inst✝⁵ : CancelMonoidWithZero M✝
                     inst✝⁴ : Finite M✝
                     M : Type u_2
                     inst✝³ : CancelMonoidWithZero M
                     inst✝² : DecidableEq M
                     inst✝¹ : Fintype M
                     inst✝ : Nontrivial M
                     ⊢ Eq (Inv.inv 0) 0
                   -/
      /-
        M✝ : Type u_1
        inst✝⁵ : CancelMonoidWithZero M✝
        inst✝⁴ : Finite M✝
        M : Type u_2
        inst✝³ : CancelMonoidWithZero M
        inst✝² : DecidableEq M
        inst✝¹ : Fintype M
        inst✝ : Nontrivial M
        a : M
        ha : Ne a 0
        ⊢ Eq (HMul.hMul a (Fintype.bijInv ⋯ 1)) 1
      -/
                   /-
                     🎉 no goals
                   -/
      exact Fintype.rightInverse_bijInv _ _
      /-
        🎉 no goals
      -/
    inv_zero := by simp [Inv.inv, dif_pos rfl] }


theorem exists_eq_pow_of_mul_eq_pow_of_coprime {R : Type*} [CommSemiring R] [IsDomain R]
    [GCDMonoid R] [Subsingleton Rˣ] {a b c : R} {n : ℕ} (cp : IsCoprime a b) (h : a * b = c ^ n) :
    ∃ d : R, a = d ^ n := by
  /-
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : GCDMonoid R
    inst✝ : Subsingleton (Units R)
    a b c : R
    n : Nat
    cp : IsCoprime a b
    h : Eq (HMul.hMul a b) (HPow.hPow c n)
    ⊢ Exists fun d => Eq a (HPow.hPow d n)
  -/
  refine exists_eq_pow_of_mul_eq_pow (isUnit_of_dvd_one ?_) h
  /-
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : GCDMonoid R
    inst✝ : Subsingleton (Units R)
    a b c : R
    n : Nat
    cp : IsCoprime a b
    h : Eq (HMul.hMul a b) (HPow.hPow c n)
    ⊢ Dvd.dvd (GCDMonoid.gcd a b) 1
  -/
  obtain ⟨x, y, hxy⟩ := cp
  /-
    case intro.intro
    R : Type u_2
    inst✝³ : CommSemiring R
    inst✝² : IsDomain R
    inst✝¹ : GCDMonoid R
    inst✝ : Subsingleton (Units R)
    a b c : R
    n : Nat
    h : Eq (HMul.hMul a b) (HPow.hPow c n)
    x y : R
    hxy : Eq (HAdd.hAdd (HMul.hMul x a) (HMul.hMul y b)) 1
    ⊢ Dvd.dvd (GCDMonoid.gcd a b) 1
  -/
  rw [← hxy]
  exact  -- Porting note: added `GCDMonoid.` twice
    dvd_add (dvd_mul_of_dvd_right (GCDMonoid.gcd_dvd_left _ _) _)
      (dvd_mul_of_dvd_right (GCDMonoid.gcd_dvd_right _ _) _)


nonrec
theorem Finset.exists_eq_pow_of_mul_eq_pow_of_coprime {ι R : Type*} [CommSemiring R] [IsDomain R]
    [GCDMonoid R] [Subsingleton Rˣ] {n : ℕ} {c : R} {s : Finset ι} {f : ι → R}
    (h : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → IsCoprime (f i) (f j))
    (hprod : ∏ i ∈ s, f i = c ^ n) : ∀ i ∈ s, ∃ d : R, f i = d ^ n := by
  classical
    intro i hi
    rw [← insert_erase hi, prod_insert (not_mem_erase i s)] at hprod
    refine
      exists_eq_pow_of_mul_eq_pow_of_coprime
        (IsCoprime.prod_right fun j hj => h i hi j (erase_subset i s hj) fun hij => ?_) hprod
    rw [hij] at hj
    exact (s.not_mem_erase _) hj


/-- Every finite domain is a division ring. More generally, they are fields; this can be found in
`Mathlib.RingTheory.LittleWedderburn`. -/
def Fintype.divisionRingOfIsDomain (R : Type*) [Ring R] [IsDomain R] [DecidableEq R] [Fintype R] :
    DivisionRing R where
  __ := Fintype.groupWithZeroOfCancel R
  __ := ‹Ring R›
  nnqsmul := _
  nnqsmul_def := fun _ _ => rfl
  qsmul := _
  qsmul_def := fun _ _ => rfl


/-- Every finite commutative domain is a field. More generally, commutativity is not required: this
can be found in `Mathlib.RingTheory.LittleWedderburn`. -/
def Fintype.fieldOfDomain (R) [CommRing R] [IsDomain R] [DecidableEq R] [Fintype R] : Field R :=
  { Fintype.divisionRingOfIsDomain R, ‹CommRing R› with }


theorem Finite.isField_of_domain (R) [CommRing R] [IsDomain R] [Finite R] : IsField R := by
  /-
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Finite R
    ⊢ IsField R
  -/
  cases nonempty_fintype R
  /-
    case intro
    R : Type u_3
    inst✝² : CommRing R
    inst✝¹ : IsDomain R
    inst✝ : Finite R
    val✝ : Fintype R
    ⊢ IsField R
  -/
  exact @Field.toIsField R (@Fintype.fieldOfDomain R _ _ (Classical.decEq R) _)
  /-
    🎉 no goals
  -/


theorem card_nthRoots_subgroup_units [Fintype G] [DecidableEq G] (f : G →* R) (hf : Injective f)
    {n : ℕ} (hn : 0 < n) (g₀ : G) :
    #{g | g ^ n = g₀} ≤ Multiset.card (nthRoots n (f g₀)) := by
  /-
    R : Type u_1
    G : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Group G
    inst✝¹ : Fintype G
    inst✝ : DecidableEq G
    f : MonoidHom G R
    hf : Function.Injective ⇑f
    n : Nat
    hn : LT.lt 0 n
    g₀ : G
    ⊢ LE.le (Finset.filter (fun g => Eq (HPow.hPow g n) g₀) Finset.univ).card (Pol …
  -/
  haveI : DecidableEq R := Classical.decEq _
  calc
    _ ≤ #(nthRoots n (f g₀)).toFinset := card_le_card_of_injOn f (by aesop) hf.injOn
    _ ≤ _ := (nthRoots n (f g₀)).toFinset_card_le


/-- A finite subgroup of the unit group of an integral domain is cyclic. -/
theorem isCyclic_of_subgroup_isDomain [Finite G] (f : G →* R) (hf : Injective f) : IsCyclic G := by
  classical
    cases nonempty_fintype G
    apply isCyclic_of_card_pow_eq_one_le
    intro n hn
    exact le_trans (card_nthRoots_subgroup_units f hf hn 1) (card_nthRoots n (f 1))


/-- The unit group of a finite integral domain is cyclic.

To support `ℤˣ` and other infinite monoids with finite groups of units, this requires only
`Finite Rˣ` rather than deducing it from `Finite R`. -/
instance [Finite Rˣ] : IsCyclic Rˣ :=
  isCyclic_of_subgroup_isDomain (Units.coeHom R) <| Units.ext


/-- A finite subgroup of the units of an integral domain is cyclic. -/
instance subgroup_units_cyclic : IsCyclic S := by
  -- Porting note: the original proof used a `coe`, but I was not able to get it to work.
  /-
    R : Type u_1
    G : Type u_2
    inst✝³ : CommRing R
    inst✝² : IsDomain R
    inst✝¹ : Group G
    S : Subgroup (Units R)
    inst✝ : Finite (Subtype fun x => Membership.mem S x)
    ⊢ IsCyclic (Subtype fun x => Membership.mem S x)
  -/
  apply isCyclic_of_subgroup_isDomain (R := R) (G := S) _ _
    /-
      R : Type u_1
      G : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : Group G
      S : Subgroup (Units R)
      inst✝ : Finite (Subtype fun x => Membership.mem S x)
      ⊢ MonoidHom (Subtype fun x => Membership.mem S x) R
    -/
  · exact MonoidHom.mk (OneHom.mk (fun s => ↑s.val) rfl) (by simp)
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      G : Type u_2
      inst✝³ : CommRing R
      inst✝² : IsDomain R
      inst✝¹ : Group G
      S : Subgroup (Units R)
      inst✝ : Finite (Subtype fun x => Membership.mem S x)
      ⊢ Function.Injective ⇑{ toFun := fun s => ↑↑s, map_one' := ⋯, map_mul' := ⋯ }
    -/
  · exact Units.ext.comp Subtype.val_injective
    /-
      🎉 no goals
    -/


theorem div_eq_quo_add_rem_div (f : R[X]) {g : R[X]} (hg : g.Monic) :
    ∃ q r : R[X], r.degree < g.degree ∧
      (algebraMap R[X] K f) / (algebraMap R[X] K g) =
        algebraMap R[X] K q + (algebraMap R[X] K r) / (algebraMap R[X] K g) := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type
    inst✝² : Field K
    inst✝¹ : Algebra (Polynomial R) K
    inst✝ : IsFractionRing (Polynomial R) K
    f g : Polynomial R
    hg : g.Monic
    ⊢ Exists fun q => Exists fun r => And (LT.lt r.degree g.degree) (Eq (HDiv.hDiv …
  -/
  refine ⟨f /ₘ g, f %ₘ g, ?_, ?_⟩
    /-
      case refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type
      inst✝² : Field K
      inst✝¹ : Algebra (Polynomial R) K
      inst✝ : IsFractionRing (Polynomial R) K
      f g : Polynomial R
      hg : g.Monic
      ⊢ LT.lt (f.modByMonic g).degree g.degree
    -/
  · exact degree_modByMonic_lt _ hg
    /-
      🎉 no goals
    -/
  · have hg' : algebraMap R[X] K g ≠ 0 :=
      -- Porting note: the proof was `by exact_mod_cast Monic.ne_zero hg`
      (map_ne_zero_iff _ (IsFractionRing.injective R[X] K)).mpr (Monic.ne_zero hg)
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type
      inst✝² : Field K
      inst✝¹ : Algebra (Polynomial R) K
      inst✝ : IsFractionRing (Polynomial R) K
      f g : Polynomial R
      hg : g.Monic
      hg' : Ne ((algebraMap (Polynomial R) K) g) 0
      ⊢ Eq (HDiv.hDiv ((algebraMap (Polynomial R) K) f) ((algebraMap (Polynomial R)  …
    -/
    field_simp [hg']
    -- Porting note: `norm_cast` was here, but does nothing.
    /-
      case refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type
      inst✝² : Field K
      inst✝¹ : Algebra (Polynomial R) K
      inst✝ : IsFractionRing (Polynomial R) K
      f g : Polynomial R
      hg : g.Monic
      hg' : Ne ((algebraMap (Polynomial R) K) g) 0
      ⊢ Eq ((algebraMap (Polynomial R) K) f) (HAdd.hAdd (HMul.hMul ((algebraMap (Pol …
    -/
    rw [add_comm, mul_comm, ← map_mul, ← map_add, modByMonic_add_div f hg]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-06-10")]
alias card_fiber_eq_of_mem_range := MonoidHom.card_fiber_eq_of_mem_range


/-- In an integral domain, a sum indexed by a nontrivial homomorphism from a finite group is zero.
-/
theorem sum_hom_units_eq_zero (f : G →* R) (hf : f ≠ 1) : ∑ g : G, f g = 0 := by
  classical
    obtain ⟨x, hx⟩ : ∃ x : MonoidHom.range f.toHomUnits,
        ∀ y : MonoidHom.range f.toHomUnits, y ∈ Submonoid.powers x :=
      IsCyclic.exists_monoid_generator
    have hx1 : x ≠ 1 := by
      rintro rfl
      apply hf
      ext g
      rw [MonoidHom.one_apply]
      cases' hx ⟨f.toHomUnits g, g, rfl⟩ with n hn
      rwa [Subtype.ext_iff, Units.ext_iff, Subtype.coe_mk, MonoidHom.coe_toHomUnits, one_pow,
        eq_comm] at hn
    replace hx1 : (x.val : R) - 1 ≠ 0 := -- Porting note: was `(x : R)`
      fun h => hx1 (Subtype.eq (Units.ext (sub_eq_zero.1 h)))
    let c := #{g | f.toHomUnits g = 1}
    calc
      ∑ g : G, f g = ∑ g : G, (f.toHomUnits g : R) := rfl
      _ = ∑ u ∈ univ.image f.toHomUnits, #{g | f.toHomUnits g = u} • (u : R) :=
        sum_comp ((↑) : Rˣ → R) f.toHomUnits
      _ = ∑ u ∈ univ.image f.toHomUnits, c • (u : R) :=
        (sum_congr rfl fun u hu => congr_arg₂ _ ?_ rfl)
      -- remaining goal 1, proven below
      -- Porting note: have to change `(b : R)` into `((b : Rˣ) : R)`
      _ = ∑ b : MonoidHom.range f.toHomUnits, c • ((b : Rˣ) : R) :=
        (Finset.sum_subtype _ (by simp) _)
      _ = c • ∑ b : MonoidHom.range f.toHomUnits, ((b : Rˣ) : R) := smul_sum.symm
      _ = c • (0 : R) := congr_arg₂ _ rfl ?_
      -- remaining goal 2, proven below
      _ = (0 : R) := smul_zero _
    · -- remaining goal 1
      show #{g : G | f.toHomUnits g = u} = c
      apply MonoidHom.card_fiber_eq_of_mem_range f.toHomUnits
      · simpa only [mem_image, mem_univ, true_and, Set.mem_range] using hu
      · exact ⟨1, f.toHomUnits.map_one⟩
    -- remaining goal 2
    show (∑ b : MonoidHom.range f.toHomUnits, ((b : Rˣ) : R)) = 0
    calc
      (∑ b : MonoidHom.range f.toHomUnits, ((b : Rˣ) : R))
        = ∑ n ∈ range (orderOf x), ((x : Rˣ) : R) ^ n :=
        Eq.symm <|
          sum_nbij (x ^ ·) (by simp only [mem_univ, forall_true_iff])
            (by simpa using pow_injOn_Iio_orderOf)
            (fun b _ => let ⟨n, hn⟩ := hx b
              ⟨n % orderOf x, mem_range.2 (Nat.mod_lt _ (orderOf_pos _)),
               -- Porting note: have to use `dsimp` to apply the function
               by dsimp at hn ⊢; rw [pow_mod_orderOf, hn]⟩)
            (by simp only [imp_true_iff, eq_self_iff_true, Subgroup.coe_pow,
                Units.val_pow_eq_pow_val])
      _ = 0 := ?_

    rw [← mul_left_inj' hx1, zero_mul, geom_sum_mul]
    norm_cast
    simp [pow_orderOf_eq_one]


/-- In an integral domain, a sum indexed by a homomorphism from a finite group is zero,
unless the homomorphism is trivial, in which case the sum is equal to the cardinality of the group.
-/
theorem sum_hom_units (f : G →* R) [Decidable (f = 1)] :
    ∑ g : G, f g = if f = 1 then Fintype.card G else 0 := by
  /-
    R : Type u_1
    G : Type u_2
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    inst✝² : Group G
    inst✝¹ : Fintype G
    f : MonoidHom G R
    inst✝ : Decidable (Eq f 1)
    ⊢ Eq (Finset.univ.sum fun g => f g) ↑(ite (Eq f 1) (Fintype.card G) 0)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      G : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      inst✝² : Group G
      inst✝¹ : Fintype G
      f : MonoidHom G R
      inst✝ : Decidable (Eq f 1)
      h : Eq f 1
      ⊢ Eq (Finset.univ.sum fun g => f g) ↑(Fintype.card G)
    -/
  · simp [h, card_univ]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      G : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      inst✝² : Group G
      inst✝¹ : Fintype G
      f : MonoidHom G R
      inst✝ : Decidable (Eq f 1)
      h : Not (Eq f 1)
      ⊢ Eq (Finset.univ.sum fun g => f g) ↑0
    -/
  · rw [cast_zero] -- Porting note: added
    /-
      case neg
      R : Type u_1
      G : Type u_2
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      inst✝² : Group G
      inst✝¹ : Fintype G
      f : MonoidHom G R
      inst✝ : Decidable (Eq f 1)
      h : Not (Eq f 1)
      ⊢ Eq (Finset.univ.sum fun g => f g) 0
    -/
    exact sum_hom_units_eq_zero f h
    /-
      🎉 no goals
    -/


