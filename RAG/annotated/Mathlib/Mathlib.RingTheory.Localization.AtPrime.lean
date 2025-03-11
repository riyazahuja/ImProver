/-- The complement of a prime ideal `P ⊆ R` is a submonoid of `R`. -/
def primeCompl : Submonoid R where
  carrier := (Pᶜ : Set R)
                 /-
                   R : Type u_1
                   inst✝³ : CommSemiring R
                   S : Type u_2
                   inst✝² : CommSemiring S
                   inst✝¹ : Algebra R S
                   P✝ : Type u_3
                   inst✝ : CommSemiring P✝
                   P : Ideal R
                   hp : P.IsPrime
                   ⊢ Membership.mem { carrier := HasCompl.compl ↑P, mul_mem' := ⋯ }.carrier 1
                 -/
  one_mem' := by convert P.ne_top_iff_one.1 hp.1
                 /-
                   🎉 no goals
                 -/
  mul_mem' {_ _} hnx hny hxy := Or.casesOn (hp.mem_or_mem hxy) hnx hny


theorem primeCompl_le_nonZeroDivisors [NoZeroDivisors R] : P.primeCompl ≤ nonZeroDivisors R :=
  le_nonZeroDivisors_of_noZeroDivisors <| not_not_intro P.zero_mem


/-- Given a prime ideal `P`, the typeclass `IsLocalization.AtPrime S P` states that `S` is
isomorphic to the localization of `R` at the complement of `P`. -/
protected abbrev IsLocalization.AtPrime :=
  IsLocalization P.primeCompl S


/-- Given a prime ideal `P`, `Localization.AtPrime P` is a localization of
`R` at the complement of `P`, as a quotient type. -/
protected abbrev Localization.AtPrime :=
  Localization P.primeCompl


theorem AtPrime.Nontrivial [IsLocalization.AtPrime S P] : Nontrivial S :=
  nontrivial_of_ne (0 : S) 1 fun hze => by
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      P : Ideal R
      hp : P.IsPrime
      inst✝ : IsLocalization.AtPrime S P
      hze : Eq 0 1
      ⊢ False
    -/
    rw [← (algebraMap R S).map_one, ← (algebraMap R S).map_zero] at hze
    /-
      R : Type u_1
      inst✝³ : CommSemiring R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      P : Ideal R
      hp : P.IsPrime
      inst✝ : IsLocalization.AtPrime S P
      hze : Eq ((algebraMap R S) 0) ((algebraMap R S) 1)
      ⊢ False
    -/
    obtain ⟨t, ht⟩ := (eq_iff_exists P.primeCompl S).1 hze
    /-
      case intro
      R : Type u_1
      inst✝³ : CommSemiring R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      P : Ideal R
      hp : P.IsPrime
      inst✝ : IsLocalization.AtPrime S P
      hze : Eq ((algebraMap R S) 0) ((algebraMap R S) 1)
      t : Subtype fun x => Membership.mem P.primeCompl x
      ht : Eq (HMul.hMul (↑t) 0) (HMul.hMul (↑t) 1)
      ⊢ False
    -/
    have htz : (t : R) = 0 := by simpa using ht.symm
    /-
      case intro
      R : Type u_1
      inst✝³ : CommSemiring R
      S : Type u_2
      inst✝² : CommSemiring S
      inst✝¹ : Algebra R S
      P : Ideal R
      hp : P.IsPrime
      inst✝ : IsLocalization.AtPrime S P
      hze : Eq ((algebraMap R S) 0) ((algebraMap R S) 1)
      t : Subtype fun x => Membership.mem P.primeCompl x
      ht : Eq (HMul.hMul (↑t) 0) (HMul.hMul (↑t) 1)
      htz : Eq (↑t) 0
      ⊢ False
    -/
    exact t.2 (htz.symm ▸ P.zero_mem : ↑t ∈ P)
    /-
      🎉 no goals
    -/


theorem AtPrime.isLocalRing [IsLocalization.AtPrime S P] : IsLocalRing S :=
  -- Porting note: since I couldn't get local instance running, I just specify it manually
  letI := AtPrime.Nontrivial S P
  IsLocalRing.of_nonunits_add
    (by
      /-
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        ⊢ ∀ (a b : S), Membership.mem (nonunits S) a → Membership.mem (nonunits S) b → …
      -/
      intro x y hx hy hu
      /-
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hx : Membership.mem (nonunits S) x
        hy : Membership.mem (nonunits S) y
        hu : IsUnit (HAdd.hAdd x y)
        ⊢ False
      -/
      cases' isUnit_iff_exists_inv.1 hu with z hxyz
      have : ∀ {r : R} {s : P.primeCompl}, mk' S r s ∈ nonunits S → r ∈ P := fun {r s} =>
        not_imp_comm.1 fun nr => isUnit_iff_exists_inv.2 ⟨mk' S ↑s (⟨r, nr⟩ : P.primeCompl),
          mk'_mul_mk'_eq_one' _ _ <| show r ∈ P.primeCompl from nr⟩
      /-
        case intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hx : Membership.mem (nonunits S) x
        hy : Membership.mem (nonunits S) y
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        hxyz : Eq (HMul.hMul (HAdd.hAdd x y) z) 1
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        ⊢ False
      -/
      rcases mk'_surjective P.primeCompl x with ⟨rx, sx, hrx⟩
      /-
        case intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hx : Membership.mem (nonunits S) x
        hy : Membership.mem (nonunits S) y
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        hxyz : Eq (HMul.hMul (HAdd.hAdd x y) z) 1
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ⊢ False
      -/
      rcases mk'_surjective P.primeCompl y with ⟨ry, sy, hry⟩
      /-
        case intro.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hx : Membership.mem (nonunits S) x
        hy : Membership.mem (nonunits S) y
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        hxyz : Eq (HMul.hMul (HAdd.hAdd x y) z) 1
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ry : R
        sy : Subtype fun x => Membership.mem P.primeCompl x
        hry : Eq (IsLocalization.mk' S ry sy) y
        ⊢ False
      -/
      rcases mk'_surjective P.primeCompl z with ⟨rz, sz, hrz⟩
      /-
        case intro.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hx : Membership.mem (nonunits S) x
        hy : Membership.mem (nonunits S) y
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        hxyz : Eq (HMul.hMul (HAdd.hAdd x y) z) 1
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ry : R
        sy : Subtype fun x => Membership.mem P.primeCompl x
        hry : Eq (IsLocalization.mk' S ry sy) y
        rz : R
        sz : Subtype fun x => Membership.mem P.primeCompl x
        hrz : Eq (IsLocalization.mk' S rz sz) z
        ⊢ False
      -/
      rw [← hrx, ← hry, ← hrz, ← mk'_add, ← mk'_mul, ← mk'_self S P.primeCompl.one_mem] at hxyz
      /-
        case intro.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hx : Membership.mem (nonunits S) x
        hy : Membership.mem (nonunits S) y
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ry : R
        sy : Subtype fun x => Membership.mem P.primeCompl x
        hry : Eq (IsLocalization.mk' S ry sy) y
        rz : R
        sz : Subtype fun x => Membership.mem P.primeCompl x
        hxyz : Eq (IsLocalization.mk' S (HMul.hMul (HAdd.hAdd (HMul.hMul rx ↑sy) (HMul …
        hrz : Eq (IsLocalization.mk' S rz sz) z
        ⊢ False
      -/
      rw [← hrx] at hx
      /-
        case intro.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hy : Membership.mem (nonunits S) y
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hx : Membership.mem (nonunits S) (IsLocalization.mk' S rx sx)
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ry : R
        sy : Subtype fun x => Membership.mem P.primeCompl x
        hry : Eq (IsLocalization.mk' S ry sy) y
        rz : R
        sz : Subtype fun x => Membership.mem P.primeCompl x
        hxyz : Eq (IsLocalization.mk' S (HMul.hMul (HAdd.hAdd (HMul.hMul rx ↑sy) (HMul …
        hrz : Eq (IsLocalization.mk' S rz sz) z
        ⊢ False
      -/
      rw [← hry] at hy
      /-
        case intro.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hx : Membership.mem (nonunits S) (IsLocalization.mk' S rx sx)
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ry : R
        sy : Subtype fun x => Membership.mem P.primeCompl x
        hy : Membership.mem (nonunits S) (IsLocalization.mk' S ry sy)
        hry : Eq (IsLocalization.mk' S ry sy) y
        rz : R
        sz : Subtype fun x => Membership.mem P.primeCompl x
        hxyz : Eq (IsLocalization.mk' S (HMul.hMul (HAdd.hAdd (HMul.hMul rx ↑sy) (HMul …
        hrz : Eq (IsLocalization.mk' S rz sz) z
        ⊢ False
      -/
      obtain ⟨t, ht⟩ := IsLocalization.eq.1 hxyz
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hx : Membership.mem (nonunits S) (IsLocalization.mk' S rx sx)
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ry : R
        sy : Subtype fun x => Membership.mem P.primeCompl x
        hy : Membership.mem (nonunits S) (IsLocalization.mk' S ry sy)
        hry : Eq (IsLocalization.mk' S ry sy) y
        rz : R
        sz : Subtype fun x => Membership.mem P.primeCompl x
        hxyz : Eq (IsLocalization.mk' S (HMul.hMul (HAdd.hAdd (HMul.hMul rx ↑sy) (HMul …
        hrz : Eq (IsLocalization.mk' S rz sz) z
        t : Subtype fun x => Membership.mem P.primeCompl x
        ht : Eq (HMul.hMul (↑t) (HMul.hMul (↑⟨1, ⋯⟩) (HMul.hMul (HAdd.hAdd (HMul.hMul  …
        ⊢ False
      -/
      simp only [mul_one, one_mul, Submonoid.coe_mul, Subtype.coe_mk] at ht
      suffices (t : R) * (sx * sy * sz) ∈ P from
        not_or_intro (mt hp.mem_or_mem <| not_or_intro sx.2 sy.2) sz.2
          (hp.mem_or_mem <| (hp.mem_or_mem this).resolve_left t.2)
      /-
        case intro.intro.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝³ : CommSemiring R
        S : Type u_2
        inst✝² : CommSemiring S
        inst✝¹ : Algebra R S
        P : Ideal R
        hp : P.IsPrime
        inst✝ : IsLocalization.AtPrime S P
        this✝ : _root_.Nontrivial S := IsLocalization.AtPrime.Nontrivial S P
        x y : S
        hu : IsUnit (HAdd.hAdd x y)
        z : S
        this : ∀ {r : R} {s : Subtype fun x => Membership.mem P.primeCompl x}, Members …
        rx : R
        sx : Subtype fun x => Membership.mem P.primeCompl x
        hx : Membership.mem (nonunits S) (IsLocalization.mk' S rx sx)
        hrx : Eq (IsLocalization.mk' S rx sx) x
        ry : R
        sy : Subtype fun x => Membership.mem P.primeCompl x
        hy : Membership.mem (nonunits S) (IsLocalization.mk' S ry sy)
        hry : Eq (IsLocalization.mk' S ry sy) y
        rz : R
        sz : Subtype fun x => Membership.mem P.primeCompl x
        hxyz : Eq (IsLocalization.mk' S (HMul.hMul (HAdd.hAdd (HMul.hMul rx ↑sy) (HMul …
        hrz : Eq (IsLocalization.mk' S rz sz) z
        t : Subtype fun x => Membership.mem P.primeCompl x
        ht : Eq (HMul.hMul (↑t) (HMul.hMul (HAdd.hAdd (HMul.hMul rx ↑sy) (HMul.hMul ry …
        ⊢ Membership.mem P (HMul.hMul (↑t) (HMul.hMul (HMul.hMul ↑sx ↑sy) ↑sz))
      -/
      rw [← ht]
      exact
        P.mul_mem_left _ <| P.mul_mem_right _ <|
            P.add_mem (P.mul_mem_right _ <| this hx) <| P.mul_mem_right _ <| this hy)


@[deprecated (since := "2024-11-09")] alias AtPrime.localRing := AtPrime.isLocalRing


/-- The localization of `R` at the complement of a prime ideal is a local ring. -/
instance AtPrime.isLocalRing : IsLocalRing (Localization P.primeCompl) :=
  IsLocalization.AtPrime.isLocalRing (Localization P.primeCompl) P


/-- The localization of an integral domain at the complement of a prime ideal is an integral domain.
-/
instance isDomain_of_local_atPrime {P : Ideal A} (_ : P.IsPrime) :
    IsDomain (Localization.AtPrime P) :=
  isDomain_localization P.primeCompl_le_nonZeroDivisors


/-- The prime ideals in the localization of a commutative ring at a prime ideal I are in
order-preserving bijection with the prime ideals contained in I. -/
def orderIsoOfPrime : { p : Ideal S // p.IsPrime } ≃o { p : Ideal R // p.IsPrime ∧ p ≤ I } :=
  (IsLocalization.orderIsoOfPrime I.primeCompl S).trans <| .setCongr _ _ <| show setOf _ = setOf _
       /-
         R : Type u_1
         inst✝⁶ : CommSemiring R
         S : Type u_2
         inst✝⁵ : CommSemiring S
         inst✝⁴ : Algebra R S
         P : Type u_3
         inst✝³ : CommSemiring P
         A : Type u_4
         inst✝² : CommRing A
         inst✝¹ : IsDomain A
         I : Ideal R
         hI : I.IsPrime
         inst✝ : IsLocalization.AtPrime S I
         ⊢ Eq (setOf fun p => And p.IsPrime (Disjoint ↑I.primeCompl ↑p)) (setOf fun p = …
       -/
    by ext; simp [Ideal.primeCompl, ← le_compl_iff_disjoint_left]
            /-
              🎉 no goals
            -/


theorem isUnit_to_map_iff (x : R) : IsUnit ((algebraMap R S) x) ↔ x ∈ I.primeCompl :=
  ⟨fun h hx =>
    (isPrime_of_isPrime_disjoint I.primeCompl S I hI disjoint_compl_left).ne_top <|
      (Ideal.map (algebraMap R S) I).eq_top_of_isUnit_mem (Ideal.mem_map_of_mem _ hx) h,
    fun h => map_units S ⟨x, h⟩⟩

-- Can't use typeclasses to infer the `IsLocalRing` instance, so use an `optParam` instead
-- (since `IsLocalRing` is a `Prop`, there should be no unification issues.)

theorem to_map_mem_maximal_iff (x : R) (h : IsLocalRing S := isLocalRing S I) :
    algebraMap R S x ∈ IsLocalRing.maximalIdeal S ↔ x ∈ I :=
  not_iff_not.mp <| by
    simpa only [IsLocalRing.mem_maximalIdeal, mem_nonunits_iff, Classical.not_not] using
      isUnit_to_map_iff S I x


theorem comap_maximalIdeal (h : IsLocalRing S := isLocalRing S I) :
    (IsLocalRing.maximalIdeal S).comap (algebraMap R S) = I :=
                        /-
                          R : Type u_1
                          inst✝³ : CommSemiring R
                          S : Type u_2
                          inst✝² : CommSemiring S
                          inst✝¹ : Algebra R S
                          I : Ideal R
                          hI : I.IsPrime
                          inst✝ : IsLocalization.AtPrime S I
                          h : optParam (IsLocalRing S) ⋯
                          x : R
                          ⊢ Iff (Membership.mem (Ideal.comap (algebraMap R S) (IsLocalRing.maximalIdeal  …
                        -/
  Ideal.ext fun x => by simpa only [Ideal.mem_comap] using to_map_mem_maximal_iff _ I x
                        /-
                          🎉 no goals
                        -/


theorem isUnit_mk'_iff (x : R) (y : I.primeCompl) : IsUnit (mk' S x y) ↔ x ∈ I.primeCompl :=
  ⟨fun h hx => mk'_mem_iff.mpr ((to_map_mem_maximal_iff S I x).mpr hx) h, fun h =>
    isUnit_iff_exists_inv.mpr ⟨mk' S ↑y ⟨x, h⟩, mk'_mul_mk'_eq_one ⟨x, h⟩ y⟩⟩


theorem mk'_mem_maximal_iff (x : R) (y : I.primeCompl) (h : IsLocalRing S := isLocalRing S I) :
    mk' S x y ∈ IsLocalRing.maximalIdeal S ↔ x ∈ I :=
  not_iff_not.mp <| by
    simpa only [IsLocalRing.mem_maximalIdeal, mem_nonunits_iff, Classical.not_not] using
      isUnit_mk'_iff S I x y


/-- The unique maximal ideal of the localization at `I.primeCompl` lies over the ideal `I`. -/
theorem AtPrime.comap_maximalIdeal :
    Ideal.comap (algebraMap R (Localization.AtPrime I))
        (IsLocalRing.maximalIdeal (Localization I.primeCompl)) =
      I :=
  -- Porting note: need to provide full name
  IsLocalization.AtPrime.comap_maximalIdeal _ _


/-- The image of `I` in the localization at `I.primeCompl` is a maximal ideal, and in particular
it is the unique maximal ideal given by the local ring structure `AtPrime.isLocalRing` -/
theorem AtPrime.map_eq_maximalIdeal :
    Ideal.map (algebraMap R (Localization.AtPrime I)) I =
      IsLocalRing.maximalIdeal (Localization I.primeCompl) := by
  convert congr_arg (Ideal.map (algebraMap R (Localization.AtPrime I)))
  -- Porting note: `algebraMap R ...` can not be solve by unification
    (AtPrime.comap_maximalIdeal (hI := hI)).symm
  -- Porting note: can not find `hI`
  /-
    case h.e'_3
    R : Type u_1
    inst✝ : CommSemiring R
    I : Ideal R
    hI : I.IsPrime
    ⊢ Eq (IsLocalRing.maximalIdeal (Localization I.primeCompl)) (Ideal.map (algebr …
  -/
  rw [map_comap I.primeCompl]
  /-
    🎉 no goals
  -/


theorem le_comap_primeCompl_iff {J : Ideal P} [J.IsPrime] {f : R →+* P} :
    I.primeCompl ≤ J.primeCompl.comap f ↔ J.comap f ≤ I :=
  ⟨fun h x hx => by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      P : Type u_3
      inst✝¹ : CommSemiring P
      I : Ideal R
      hI : I.IsPrime
      J : Ideal P
      inst✝ : J.IsPrime
      f : RingHom R P
      h : LE.le I.primeCompl (Submonoid.comap f J.primeCompl)
      x : R
      hx : Membership.mem (Ideal.comap f J) x
      ⊢ Membership.mem I x
    -/
    contrapose! hx
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      P : Type u_3
      inst✝¹ : CommSemiring P
      I : Ideal R
      hI : I.IsPrime
      J : Ideal P
      inst✝ : J.IsPrime
      f : RingHom R P
      h : LE.le I.primeCompl (Submonoid.comap f J.primeCompl)
      x : R
      hx : Not (Membership.mem I x)
      ⊢ Not (Membership.mem (Ideal.comap f J) x)
    -/
    exact h hx,
    /-
      🎉 no goals
    -/
   fun h _ hx hfxJ => hx (h hfxJ)⟩


/-- For a ring hom `f : R →+* S` and a prime ideal `J` in `S`, the induced ring hom from the
localization of `R` at `J.comap f` to the localization of `S` at `J`.

To make this definition more flexible, we allow any ideal `I` of `R` as input, together with a proof
that `I = J.comap f`. This can be useful when `I` is not definitionally equal to `J.comap f`.
 -/
noncomputable def localRingHom (J : Ideal P) [J.IsPrime] (f : R →+* P) (hIJ : I = J.comap f) :
    Localization.AtPrime I →+* Localization.AtPrime J :=
  IsLocalization.map (Localization.AtPrime J) f (le_comap_primeCompl_iff.mpr (ge_of_eq hIJ))


theorem localRingHom_to_map (J : Ideal P) [J.IsPrime] (f : R →+* P) (hIJ : I = J.comap f)
    (x : R) : localRingHom I J f hIJ (algebraMap _ _ x) = algebraMap _ _ (f x) :=
  map_eq _ _


theorem localRingHom_mk' (J : Ideal P) [J.IsPrime] (f : R →+* P) (hIJ : I = J.comap f) (x : R)
    (y : I.primeCompl) :
    localRingHom I J f hIJ (IsLocalization.mk' _ x y) =
      IsLocalization.mk' (Localization.AtPrime J) (f x)
        (⟨f y, le_comap_primeCompl_iff.mpr (ge_of_eq hIJ) y.2⟩ : J.primeCompl) :=
  map_mk' _ _ _


@[instance]
theorem isLocalHom_localRingHom (J : Ideal P) [hJ : J.IsPrime] (f : R →+* P)
    (hIJ : I = J.comap f) : IsLocalHom (localRingHom I J f hIJ) :=
  IsLocalHom.mk fun x hx => by
    /-
      R : Type u_1
      inst✝¹ : CommSemiring R
      P : Type u_3
      inst✝ : CommSemiring P
      I : Ideal R
      hI : I.IsPrime
      J : Ideal P
      hJ : J.IsPrime
      f : RingHom R P
      hIJ : Eq I (Ideal.comap f J)
      x : Localization.AtPrime I
      hx : IsUnit ((Localization.localRingHom I J f hIJ) x)
      ⊢ IsUnit x
    -/
    rcases IsLocalization.mk'_surjective I.primeCompl x with ⟨r, s, rfl⟩
    /-
      case intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      P : Type u_3
      inst✝ : CommSemiring P
      I : Ideal R
      hI : I.IsPrime
      J : Ideal P
      hJ : J.IsPrime
      f : RingHom R P
      hIJ : Eq I (Ideal.comap f J)
      r : R
      s : Subtype fun x => Membership.mem I.primeCompl x
      hx : IsUnit ((Localization.localRingHom I J f hIJ) (IsLocalization.mk' (Locali …
      ⊢ IsUnit (IsLocalization.mk' (Localization.AtPrime I) r s)
    -/
    rw [localRingHom_mk'] at hx
    /-
      case intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      P : Type u_3
      inst✝ : CommSemiring P
      I : Ideal R
      hI : I.IsPrime
      J : Ideal P
      hJ : J.IsPrime
      f : RingHom R P
      hIJ : Eq I (Ideal.comap f J)
      r : R
      s : Subtype fun x => Membership.mem I.primeCompl x
      hx : IsUnit (IsLocalization.mk' (Localization.AtPrime J) (f r) ⟨f ↑s, ⋯⟩)
      ⊢ IsUnit (IsLocalization.mk' (Localization.AtPrime I) r s)
    -/
    rw [AtPrime.isUnit_mk'_iff] at hx ⊢
    /-
      case intro.intro
      R : Type u_1
      inst✝¹ : CommSemiring R
      P : Type u_3
      inst✝ : CommSemiring P
      I : Ideal R
      hI : I.IsPrime
      J : Ideal P
      hJ : J.IsPrime
      f : RingHom R P
      hIJ : Eq I (Ideal.comap f J)
      r : R
      s : Subtype fun x => Membership.mem I.primeCompl x
      hx : Membership.mem J.primeCompl (f r)
      ⊢ Membership.mem I.primeCompl r
    -/
    exact fun hr => hx ((SetLike.ext_iff.mp hIJ r).mp hr)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-10")]
alias isLocalRingHom_localRingHom := isLocalHom_localRingHom


theorem localRingHom_unique (J : Ideal P) [J.IsPrime] (f : R →+* P) (hIJ : I = J.comap f)
    {j : Localization.AtPrime I →+* Localization.AtPrime J}
    (hj : ∀ x : R, j (algebraMap _ _ x) = algebraMap _ _ (f x)) : localRingHom I J f hIJ = j :=
  map_unique _ _ hj


@[simp]
theorem localRingHom_id : localRingHom I I (RingHom.id R) (Ideal.comap_id I).symm = RingHom.id _ :=
  localRingHom_unique _ _ _ _ fun _ => rfl

-- Porting note: simplifier won't pick up this lemma, so deleted @[simp]

theorem localRingHom_comp {S : Type*} [CommSemiring S] (J : Ideal S) [hJ : J.IsPrime] (K : Ideal P)
    [hK : K.IsPrime] (f : R →+* S) (hIJ : I = J.comap f) (g : S →+* P) (hJK : J = K.comap g) :
                                    /-
                                      R : Type u_1
                                      inst✝⁴ : CommSemiring R
                                      S✝ : Type u_2
                                      inst✝³ : CommSemiring S✝
                                      inst✝² : Algebra R S✝
                                      P : Type u_3
                                      inst✝¹ : CommSemiring P
                                      I : Ideal R
                                      hI : I.IsPrime
                                      S : Type u_4
                                      inst✝ : CommSemiring S
                                      J : Ideal S
                                      hJ : J.IsPrime
                                      K : Ideal P
                                      hK : K.IsPrime
                                      f : RingHom R S
                                      hIJ : Eq I (Ideal.comap f J)
                                      g : RingHom S P
                                      hJK : Eq J (Ideal.comap g K)
                                      ⊢ Eq I (Ideal.comap (g.comp f) K)
                                    -/
    localRingHom I K (g.comp f) (by rw [hIJ, hJK, Ideal.comap_comap f g]) =
                                    /-
                                      🎉 no goals
                                    -/
      (localRingHom J K g hJK).comp (localRingHom I J f hIJ) :=
  localRingHom_unique _ _ _ _ fun r => by
    /-
      R : Type u_1
      inst✝² : CommSemiring R
      P : Type u_3
      inst✝¹ : CommSemiring P
      I : Ideal R
      hI : I.IsPrime
      S : Type u_4
      inst✝ : CommSemiring S
      J : Ideal S
      hJ : J.IsPrime
      K : Ideal P
      hK : K.IsPrime
      f : RingHom R S
      hIJ : Eq I (Ideal.comap f J)
      g : RingHom S P
      hJK : Eq J (Ideal.comap g K)
      r : R
      ⊢ Eq (((Localization.localRingHom J K g hJK).comp (Localization.localRingHom I …
    -/
    simp only [Function.comp_apply, RingHom.coe_comp, localRingHom_to_map]
    /-
      🎉 no goals
    -/


/-- `Localization.localRingHom` specialized to a projection homomorphism from a product ring. -/
noncomputable abbrev mapPiEvalRingHom :
    Localization.AtPrime (I.comap <| Pi.evalRingHom R i) →+* Localization.AtPrime I :=
  localRingHom _ _ _ rfl


theorem mapPiEvalRingHom_bijective : Function.Bijective (mapPiEvalRingHom I) :=
  Localization.mapPiEvalRingHom_bijective _


theorem mapPiEvalRingHom_comp_algebraMap :
    (mapPiEvalRingHom I).comp (algebraMap _ _) = (algebraMap _ _).comp (Pi.evalRingHom R i) :=
  IsLocalization.map_comp _


theorem mapPiEvalRingHom_algebraMap_apply {r : Π i, R i} :
    mapPiEvalRingHom I (algebraMap _ _ r) = algebraMap _ _ (r i) :=
  localRingHom_to_map ..


