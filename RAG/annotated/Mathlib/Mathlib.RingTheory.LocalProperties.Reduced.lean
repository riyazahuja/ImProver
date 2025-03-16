/-- `M⁻¹R` is reduced if `R` is reduced. -/
theorem isReduced_localizationPreserves : LocalizationPreserves fun R _ => IsReduced R := by
  /-
    ⊢ LocalizationPreserves fun R x => IsReduced R
  -/
  introv R _ _
  /-
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    ⊢ IsReduced S
  -/
  constructor
  /-
    case eq_zero
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    ⊢ ∀ (x : S), IsNilpotent x → Eq x 0
  -/
  rintro x ⟨_ | n, e⟩
    /-
      case eq_zero.intro.zero
      R : Type u_1
      hR : CommRing R
      M : Submonoid R
      S : Type u_1
      hS : CommRing S
      inst✝¹ : Algebra R S
      inst✝ : IsLocalization M S
      a✝ : IsReduced R
      x : S
      e : Eq (HPow.hPow x 0) 0
      ⊢ Eq x 0
    -/
  · simpa using congr_arg (· * x) e
    /-
      🎉 no goals
    -/
  /-
    case eq_zero.intro.succ
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    ⊢ Eq x 0
  -/
  obtain ⟨⟨y, m⟩, hx⟩ := IsLocalization.surj M x
  /-
    case eq_zero.intro.succ.intro.mk
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑{ fst := y, snd := m }.2)) ((algebraMa …
    ⊢ Eq x 0
  -/
  dsimp only at hx
  /-
    case eq_zero.intro.succ.intro.mk
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    ⊢ Eq x 0
  -/
  let hx' := congr_arg (· ^ n.succ) hx
  /-
    case eq_zero.intro.succ.intro.mk
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq ((fun x => HPow.hPow x n.succ) (HMul.hMul x ((algebraMap R S) ↑m))) ( …
    ⊢ Eq x 0
  -/
  simp only [mul_pow, e, zero_mul, ← RingHom.map_pow] at hx'
  /-
    case eq_zero.intro.succ.intro.mk
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq 0 ((algebraMap R S) (HPow.hPow y n.succ))
    ⊢ Eq x 0
  -/
  rw [← (algebraMap R S).map_zero] at hx'
  /-
    case eq_zero.intro.succ.intro.mk
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq ((algebraMap R S) 0) ((algebraMap R S) (HPow.hPow y n.succ))
    ⊢ Eq x 0
  -/
  obtain ⟨m', hm'⟩ := (IsLocalization.eq_iff_exists M S).mp hx'
  /-
    case eq_zero.intro.succ.intro.mk.intro
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq ((algebraMap R S) 0) ((algebraMap R S) (HPow.hPow y n.succ))
    m' : Subtype fun x => Membership.mem M x
    hm' : Eq (HMul.hMul (↑m') 0) (HMul.hMul (↑m') (HPow.hPow y n.succ))
    ⊢ Eq x 0
  -/
  apply_fun (· * (m' : R) ^ n) at hm'
  /-
    case eq_zero.intro.succ.intro.mk.intro
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq ((algebraMap R S) 0) ((algebraMap R S) (HPow.hPow y n.succ))
    m' : Subtype fun x => Membership.mem M x
    hm' : Eq (HMul.hMul (HMul.hMul (↑m') 0) (HPow.hPow (↑m') n)) (HMul.hMul (HMul. …
    ⊢ Eq x 0
  -/
  simp only [mul_assoc, zero_mul, mul_zero] at hm'
  /-
    case eq_zero.intro.succ.intro.mk.intro
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq ((algebraMap R S) 0) ((algebraMap R S) (HPow.hPow y n.succ))
    m' : Subtype fun x => Membership.mem M x
    hm' : Eq 0 (HMul.hMul (↑m') (HMul.hMul (HPow.hPow y n.succ) (HPow.hPow (↑m') n …
    ⊢ Eq x 0
  -/
  rw [← mul_left_comm, ← pow_succ', ← mul_pow] at hm'
  /-
    case eq_zero.intro.succ.intro.mk.intro
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq ((algebraMap R S) 0) ((algebraMap R S) (HPow.hPow y n.succ))
    m' : Subtype fun x => Membership.mem M x
    hm' : Eq 0 (HPow.hPow (HMul.hMul y ↑m') n.succ)
    ⊢ Eq x 0
  -/
  replace hm' := IsNilpotent.eq_zero ⟨_, hm'.symm⟩
  rw [← (IsLocalization.map_units S m).mul_left_inj, hx, zero_mul,
    IsLocalization.map_eq_zero_iff M]
  /-
    case eq_zero.intro.succ.intro.mk.intro
    R : Type u_1
    hR : CommRing R
    M : Submonoid R
    S : Type u_1
    hS : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    a✝ : IsReduced R
    x : S
    n : Nat
    e : Eq (HPow.hPow x (HAdd.hAdd n 1)) 0
    y : R
    m : Subtype fun x => Membership.mem M x
    hx : Eq (HMul.hMul x ((algebraMap R S) ↑m)) ((algebraMap R S) y)
    hx' : Eq ((algebraMap R S) 0) ((algebraMap R S) (HPow.hPow y n.succ))
    m' : Subtype fun x => Membership.mem M x
    hm' : Eq (HMul.hMul y ↑m') 0
    ⊢ Exists fun m => Eq (HMul.hMul (↑m) y) 0
  -/
  exact ⟨m', by rw [← hm', mul_comm]⟩
  /-
    🎉 no goals
  -/


instance {R : Type*} [CommRing R] (M : Submonoid R) [IsReduced R] : IsReduced (Localization M) :=
  isReduced_localizationPreserves M _ inferInstance


/-- `R` is reduced if `Rₘ` is reduced for all maximal ideal `m`. -/
theorem isReduced_ofLocalizationMaximal : OfLocalizationMaximal fun R _ => IsReduced R := by
  /-
    ⊢ OfLocalizationMaximal fun R x => IsReduced R
  -/
  introv R h
  /-
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (J : Ideal R) (x : J.IsMaximal), (fun R x => IsReduced R) (Localization. …
    ⊢ IsReduced R
  -/
  constructor
  /-
    case eq_zero
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (J : Ideal R) (x : J.IsMaximal), (fun R x => IsReduced R) (Localization. …
    ⊢ ∀ (x : R), IsNilpotent x → Eq x 0
  -/
  intro x hx
  /-
    case eq_zero
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (J : Ideal R) (x : J.IsMaximal), (fun R x => IsReduced R) (Localization. …
    x : R
    hx : IsNilpotent x
    ⊢ Eq x 0
  -/
  apply eq_zero_of_localization
  /-
    case eq_zero.h
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (J : Ideal R) (x : J.IsMaximal), (fun R x => IsReduced R) (Localization. …
    x : R
    hx : IsNilpotent x
    ⊢ ∀ (J : Ideal R) (x_1 : J.IsMaximal), Eq ((algebraMap R (Localization.AtPrime …
  -/
  intro J hJ
  /-
    case eq_zero.h
    R : Type u_1
    inst✝ : CommRing R
    h : ∀ (J : Ideal R) (x : J.IsMaximal), (fun R x => IsReduced R) (Localization. …
    x : R
    hx : IsNilpotent x
    J : Ideal R
    hJ : J.IsMaximal
    ⊢ Eq ((algebraMap R (Localization.AtPrime J)) x) 0
  -/
  specialize h J hJ
  /-
    case eq_zero.h
    R : Type u_1
    inst✝ : CommRing R
    x : R
    hx : IsNilpotent x
    J : Ideal R
    hJ : J.IsMaximal
    h : IsReduced (Localization.AtPrime J)
    ⊢ Eq ((algebraMap R (Localization.AtPrime J)) x) 0
  -/
  exact (hx.map <| algebraMap R <| Localization.AtPrime J).eq_zero
  /-
    🎉 no goals
  -/

