/--
A ring homomorphism `R →+* S` is surjective on stalks if `R_p →+* S_q` is surjective for all pairs
of primes `p = f⁻¹(q)`.
-/
def SurjectiveOnStalks (f : R →+* S) : Prop :=
  ∀ (P : Ideal S) (_ : P.IsPrime), Function.Surjective (Localization.localRingHom _ P f rfl)


/--
`R_p →+* S_q` is surjective if and only if
every `x : S` is of the form `f x / f r` for some `f r ∉ q`.
This is useful when proving `SurjectiveOnStalks`.
-/
lemma surjective_localRingHom_iff (P : Ideal S) [P.IsPrime] :
    Function.Surjective (Localization.localRingHom _ P f rfl) ↔
      ∀ s : S, ∃ x r : R, ∃ c ∉ P, f r ∉ P ∧ c * f r * s = c * f x := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    f : RingHom R S
    P : Ideal S
    inst✝ : P.IsPrime
    ⊢ Iff (Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f P) P f ⋯ …
  -/
  constructor
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      ⊢ Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f P) P f ⋯) → ∀ …
    -/
  · intro H y
    /-
      case mp
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f P) P f ⋯)
      y : S
      ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem P c …
    -/
    obtain ⟨a, ha⟩ := H (IsLocalization.mk' _ y (1 : P.primeCompl))
    /-
      case mp.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f P) P f ⋯)
      y : S
      a : Localization.AtPrime (Ideal.comap f P)
      ha : Eq ((Localization.localRingHom (Ideal.comap f P) P f ⋯) a) (IsLocalizatio …
      ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem P c …
    -/
    obtain ⟨a, t, rfl⟩ := IsLocalization.mk'_surjective (P.comap f).primeCompl a
    rw [Localization.localRingHom_mk', IsLocalization.mk'_eq_iff_eq,
      Submonoid.coe_one, one_mul, IsLocalization.eq_iff_exists P.primeCompl] at ha
    /-
      case mp.intro.intro.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f P) P f ⋯)
      y : S
      a : R
      t : Subtype fun x => Membership.mem (Ideal.comap f P).primeCompl x
      ha : Exists fun c => Eq (HMul.hMul (↑c) (f a)) (HMul.hMul (↑c) (HMul.hMul (↑⟨f …
      ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem P c …
    -/
    obtain ⟨c, hc⟩ := ha
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f P) P f ⋯)
      y : S
      a : R
      t : Subtype fun x => Membership.mem (Ideal.comap f P).primeCompl x
      c : Subtype fun x => Membership.mem P.primeCompl x
      hc : Eq (HMul.hMul (↑c) (f a)) (HMul.hMul (↑c) (HMul.hMul (↑⟨f ↑t, ⋯⟩) y))
      ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem P c …
    -/
    simp only [← mul_assoc] at hc
    /-
      case mp.intro.intro.intro.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f P) P f ⋯)
      y : S
      a : R
      t : Subtype fun x => Membership.mem (Ideal.comap f P).primeCompl x
      c : Subtype fun x => Membership.mem P.primeCompl x
      hc : Eq (HMul.hMul (↑c) (f a)) (HMul.hMul (HMul.hMul (↑c) (f ↑t)) y)
      ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem P c …
    -/
    exact ⟨_, _, _, c.2, t.2, hc.symm⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      ⊢ (∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Member …
    -/
  · refine fun H y ↦ Localization.ind (fun ⟨y, t, h⟩ ↦ ?_) y
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
      y✝ : Localization.AtPrime P
      x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
      y t : S
      h : Membership.mem P.primeCompl t
      ⊢ Exists fun a => Eq ((Localization.localRingHom (Ideal.comap f P) P f ⋯) a) ( …
    -/
    simp only
    /-
      case mpr
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
      y✝ : Localization.AtPrime P
      x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
      y t : S
      h : Membership.mem P.primeCompl t
      ⊢ Exists fun a => Eq ((Localization.localRingHom (Ideal.comap f P) P f ⋯) a) ( …
    -/
    obtain ⟨yx, ys, yc, hyc, hy, ey⟩ := H y
    /-
      case mpr.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
      y✝ : Localization.AtPrime P
      x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
      y t : S
      h : Membership.mem P.primeCompl t
      yx ys : R
      yc : S
      hyc : Not (Membership.mem P yc)
      hy : Not (Membership.mem P (f ys))
      ey : Eq (HMul.hMul (HMul.hMul yc (f ys)) y) (HMul.hMul yc (f yx))
      ⊢ Exists fun a => Eq ((Localization.localRingHom (Ideal.comap f P) P f ⋯) a) ( …
    -/
    obtain ⟨tx, ts, yt, hyt, ht, et⟩ := H t
    /-
      case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
      R : Type u_1
      inst✝² : CommRing R
      S : Type u_2
      inst✝¹ : CommRing S
      f : RingHom R S
      P : Ideal S
      inst✝ : P.IsPrime
      H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
      y✝ : Localization.AtPrime P
      x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
      y t : S
      h : Membership.mem P.primeCompl t
      yx ys : R
      yc : S
      hyc : Not (Membership.mem P yc)
      hy : Not (Membership.mem P (f ys))
      ey : Eq (HMul.hMul (HMul.hMul yc (f ys)) y) (HMul.hMul yc (f yx))
      tx ts : R
      yt : S
      hyt : Not (Membership.mem P yt)
      ht : Not (Membership.mem P (f ts))
      et : Eq (HMul.hMul (HMul.hMul yt (f ts)) t) (HMul.hMul yt (f tx))
      ⊢ Exists fun a => Eq ((Localization.localRingHom (Ideal.comap f P) P f ⋯) a) ( …
    -/
    refine ⟨Localization.mk (yx * ts) ⟨ys * tx, Submonoid.mul_mem _ hy ?_⟩, ?_⟩
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
        R : Type u_1
        inst✝² : CommRing R
        S : Type u_2
        inst✝¹ : CommRing S
        f : RingHom R S
        P : Ideal S
        inst✝ : P.IsPrime
        H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
        y✝ : Localization.AtPrime P
        x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
        y t : S
        h : Membership.mem P.primeCompl t
        yx ys : R
        yc : S
        hyc : Not (Membership.mem P yc)
        hy : Not (Membership.mem P (f ys))
        ey : Eq (HMul.hMul (HMul.hMul yc (f ys)) y) (HMul.hMul yc (f yx))
        tx ts : R
        yt : S
        hyt : Not (Membership.mem P yt)
        ht : Not (Membership.mem P (f ts))
        et : Eq (HMul.hMul (HMul.hMul yt (f ts)) t) (HMul.hMul yt (f tx))
        ⊢ Membership.mem (Ideal.comap f P).primeCompl tx
      -/
    · exact fun H ↦ mul_mem (P.primeCompl.mul_mem hyt ht) h (et ▸ Ideal.mul_mem_left _ yt H)
      /-
        🎉 no goals
      -/
    · simp only [Localization.mk_eq_mk', Localization.localRingHom_mk', map_mul f,
        IsLocalization.mk'_eq_iff_eq, IsLocalization.eq_iff_exists P.primeCompl]
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        R : Type u_1
        inst✝² : CommRing R
        S : Type u_2
        inst✝¹ : CommRing S
        f : RingHom R S
        P : Ideal S
        inst✝ : P.IsPrime
        H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
        y✝ : Localization.AtPrime P
        x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
        y t : S
        h : Membership.mem P.primeCompl t
        yx ys : R
        yc : S
        hyc : Not (Membership.mem P yc)
        hy : Not (Membership.mem P (f ys))
        ey : Eq (HMul.hMul (HMul.hMul yc (f ys)) y) (HMul.hMul yc (f yx))
        tx ts : R
        yt : S
        hyt : Not (Membership.mem P yt)
        ht : Not (Membership.mem P (f ts))
        et : Eq (HMul.hMul (HMul.hMul yt (f ts)) t) (HMul.hMul yt (f tx))
        ⊢ Exists fun c => Eq (HMul.hMul (↑c) (HMul.hMul t (HMul.hMul (f yx) (f ts))))  …
      -/
      refine ⟨⟨yc, hyc⟩ * ⟨yt, hyt⟩, ?_⟩
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        R : Type u_1
        inst✝² : CommRing R
        S : Type u_2
        inst✝¹ : CommRing S
        f : RingHom R S
        P : Ideal S
        inst✝ : P.IsPrime
        H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
        y✝ : Localization.AtPrime P
        x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
        y t : S
        h : Membership.mem P.primeCompl t
        yx ys : R
        yc : S
        hyc : Not (Membership.mem P yc)
        hy : Not (Membership.mem P (f ys))
        ey : Eq (HMul.hMul (HMul.hMul yc (f ys)) y) (HMul.hMul yc (f yx))
        tx ts : R
        yt : S
        hyt : Not (Membership.mem P yt)
        ht : Not (Membership.mem P (f ts))
        et : Eq (HMul.hMul (HMul.hMul yt (f ts)) t) (HMul.hMul yt (f tx))
        ⊢ Eq (HMul.hMul (↑(HMul.hMul ⟨yc, hyc⟩ ⟨yt, hyt⟩)) (HMul.hMul t (HMul.hMul (f  …
      -/
      simp only [Submonoid.coe_mul]
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
        R : Type u_1
        inst✝² : CommRing R
        S : Type u_2
        inst✝¹ : CommRing S
        f : RingHom R S
        P : Ideal S
        inst✝ : P.IsPrime
        H : ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Membe …
        y✝ : Localization.AtPrime P
        x✝ : Prod S (Subtype fun x => Membership.mem P.primeCompl x)
        y t : S
        h : Membership.mem P.primeCompl t
        yx ys : R
        yc : S
        hyc : Not (Membership.mem P yc)
        hy : Not (Membership.mem P (f ys))
        ey : Eq (HMul.hMul (HMul.hMul yc (f ys)) y) (HMul.hMul yc (f yx))
        tx ts : R
        yt : S
        hyt : Not (Membership.mem P yt)
        ht : Not (Membership.mem P (f ts))
        et : Eq (HMul.hMul (HMul.hMul yt (f ts)) t) (HMul.hMul yt (f tx))
        ⊢ Eq (HMul.hMul (HMul.hMul yc yt) (HMul.hMul t (HMul.hMul (f yx) (f ts)))) (HM …
      -/
                                                    /-
                                                      🎉 no goals
                                                    -/
      convert congr($(ey.symm) * $(et)) using 1 <;> ring
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma surjectiveOnStalks_iff_forall_ideal :
    f.SurjectiveOnStalks ↔
      ∀ I : Ideal S, I ≠ ⊤ → ∀ s : S, ∃ x r : R, ∃ c ∉ I, f r ∉ I ∧ c * f r * s = c * f x := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Iff f.SurjectiveOnStalks (∀ (I : Ideal S), Ne I Top.top → ∀ (s : S), Exists  …
  -/
  simp_rw [SurjectiveOnStalks, surjective_localRingHom_iff]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Iff (∀ (P : Ideal S), P.IsPrime → ∀ (s : S), Exists fun x => Exists fun r => …
  -/
  refine ⟨fun H I hI s ↦ ?_, fun H I hI ↦ H I hI.ne_top⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    H : ∀ (P : Ideal S), P.IsPrime → ∀ (s : S), Exists fun x => Exists fun r => Ex …
    I : Ideal S
    hI : Ne I Top.top
    s : S
    ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem I c …
  -/
  obtain ⟨M, hM, hIM⟩ := I.exists_le_maximal hI
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    H : ∀ (P : Ideal S), P.IsPrime → ∀ (s : S), Exists fun x => Exists fun r => Ex …
    I : Ideal S
    hI : Ne I Top.top
    s : S
    M : Ideal S
    hM : M.IsMaximal
    hIM : LE.le I M
    ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem I c …
  -/
  obtain ⟨x, r, c, hc, hr, e⟩ := H M hM.isPrime s
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    H : ∀ (P : Ideal S), P.IsPrime → ∀ (s : S), Exists fun x => Exists fun r => Ex …
    I : Ideal S
    hI : Ne I Top.top
    s : S
    M : Ideal S
    hM : M.IsMaximal
    hIM : LE.le I M
    x r : R
    c : S
    hc : Not (Membership.mem M c)
    hr : Not (Membership.mem M (f r))
    e : Eq (HMul.hMul (HMul.hMul c (f r)) s) (HMul.hMul c (f x))
    ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem I c …
  -/
  exact ⟨x, r, c, fun h ↦ hc (hIM h), fun h ↦ hr (hIM h), e⟩
  /-
    🎉 no goals
  -/


lemma surjectiveOnStalks_iff_forall_maximal :
    f.SurjectiveOnStalks ↔ ∀ (I : Ideal S) (_ : I.IsMaximal),
      Function.Surjective (Localization.localRingHom _ I f rfl) := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Iff f.SurjectiveOnStalks (∀ (I : Ideal S) (x : I.IsMaximal), Function.Surjec …
  -/
  refine ⟨fun H I hI ↦ H I hI.isPrime, fun H I hI ↦ ?_⟩
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    H : ∀ (I : Ideal S) (x : I.IsMaximal), Function.Surjective ⇑(Localization.loca …
    I : Ideal S
    hI : I.IsPrime
    ⊢ Function.Surjective ⇑(Localization.localRingHom (Ideal.comap f I) I f ⋯)
  -/
  simp_rw [surjective_localRingHom_iff] at H ⊢
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hI : I.IsPrime
    H : ∀ (I : Ideal S), I.IsMaximal → ∀ (s : S), Exists fun x => Exists fun r =>  …
    ⊢ ∀ (s : S), Exists fun x => Exists fun r => Exists fun c => And (Not (Members …
  -/
  intro s
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hI : I.IsPrime
    H : ∀ (I : Ideal S), I.IsMaximal → ∀ (s : S), Exists fun x => Exists fun r =>  …
    s : S
    ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem I c …
  -/
  obtain ⟨M, hM, hIM⟩ := I.exists_le_maximal hI.ne_top
  /-
    case intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hI : I.IsPrime
    H : ∀ (I : Ideal S), I.IsMaximal → ∀ (s : S), Exists fun x => Exists fun r =>  …
    s : S
    M : Ideal S
    hM : M.IsMaximal
    hIM : LE.le I M
    ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem I c …
  -/
  obtain ⟨x, r, c, hc, hr, e⟩ := H M hM s
  /-
    case intro.intro.intro.intro.intro.intro.intro
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    I : Ideal S
    hI : I.IsPrime
    H : ∀ (I : Ideal S), I.IsMaximal → ∀ (s : S), Exists fun x => Exists fun r =>  …
    s : S
    M : Ideal S
    hM : M.IsMaximal
    hIM : LE.le I M
    x r : R
    c : S
    hc : Not (Membership.mem M c)
    hr : Not (Membership.mem M (f r))
    e : Eq (HMul.hMul (HMul.hMul c (f r)) s) (HMul.hMul c (f x))
    ⊢ Exists fun x => Exists fun r => Exists fun c => And (Not (Membership.mem I c …
  -/
  exact ⟨x, r, c, fun h ↦ hc (hIM h), fun h ↦ hr (hIM h), e⟩
  /-
    🎉 no goals
  -/


lemma surjectiveOnStalks_iff_forall_maximal' :
    f.SurjectiveOnStalks ↔ ∀ I : Ideal S, I.IsMaximal →
      ∀ s : S, ∃ x r : R, ∃ c ∉ I, f r ∉ I ∧ c * f r * s = c * f x := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    S : Type u_2
    inst✝ : CommRing S
    f : RingHom R S
    ⊢ Iff f.SurjectiveOnStalks (∀ (I : Ideal S), I.IsMaximal → ∀ (s : S), Exists f …
  -/
  simp only [surjectiveOnStalks_iff_forall_maximal, surjective_localRingHom_iff]
  /-
    🎉 no goals
  -/


lemma surjectiveOnStalks_of_exists_div (h : ∀ x : S, ∃ r s : R, IsUnit (f s) ∧ f s * x = f r) :
    SurjectiveOnStalks f :=
  surjectiveOnStalks_iff_forall_ideal.mpr fun I hI x ↦
    let ⟨r, s, hr, hr'⟩ := h x
                 /-
                   R : Type u_1
                   inst✝¹ : CommRing R
                   S : Type u_2
                   inst✝ : CommRing S
                   f : RingHom R S
                   h : ∀ (x : S), Exists fun r => Exists fun s => And (IsUnit (f s)) (Eq (HMul.hM …
                   I : Ideal S
                   hI : Ne I Top.top
                   x : S
                   r s : R
                   hr : IsUnit (f s)
                   hr' : Eq (HMul.hMul (f s) x) (f r)
                   ⊢ Not (Membership.mem I 1)
                 -/
                 /-
                   🎉 no goals
                 -/
    ⟨r, s, 1, by simpa [← Ideal.eq_top_iff_one], fun h ↦ hI (I.eq_top_of_isUnit_mem h hr), by simpa⟩
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/


lemma surjectiveOnStalks_of_surjective (h : Function.Surjective f) :
    SurjectiveOnStalks f :=
  surjectiveOnStalks_iff_forall_ideal.mpr fun _ _ s ↦
    let ⟨r, hr⟩ := h s
                 /-
                   R : Type u_1
                   inst✝¹ : CommRing R
                   S : Type u_2
                   inst✝ : CommRing S
                   f : RingHom R S
                   h : Function.Surjective ⇑f
                   x✝¹ : Ideal S
                   x✝ : Ne x✝¹ Top.top
                   s : S
                   r : R
                   hr : Eq (f r) s
                   ⊢ Not (Membership.mem x✝¹ 1)
                 -/
                 /-
                   🎉 no goals
                 -/
                                                    /-
                                                      🎉 no goals
                                                    -/
    ⟨r, 1, 1, by simpa [← Ideal.eq_top_iff_one], by simpa [← Ideal.eq_top_iff_one], by simp [hr]⟩
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


lemma SurjectiveOnStalks.comp (hg : SurjectiveOnStalks g) (hf : SurjectiveOnStalks f) :
    SurjectiveOnStalks (g.comp f) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    T : Type u_3
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : g.SurjectiveOnStalks
    hf : f.SurjectiveOnStalks
    ⊢ (g.comp f).SurjectiveOnStalks
  -/
  intros I hI
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    T : Type u_3
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : g.SurjectiveOnStalks
    hf : f.SurjectiveOnStalks
    I : Ideal T
    hI : I.IsPrime
    ⊢ Function.Surjective ⇑(Localization.localRingHom (Ideal.comap (g.comp f) I) I …
  -/
  have := (hg I hI).comp (hf _ (hI.comap g))
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    T : Type u_3
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : g.SurjectiveOnStalks
    hf : f.SurjectiveOnStalks
    I : Ideal T
    hI : I.IsPrime
    this : Function.Surjective (Function.comp ⇑(Localization.localRingHom (Ideal.c …
    ⊢ Function.Surjective ⇑(Localization.localRingHom (Ideal.comap (g.comp f) I) I …
  -/
  rwa [← RingHom.coe_comp, ← Localization.localRingHom_comp] at this
  /-
    🎉 no goals
  -/


lemma SurjectiveOnStalks.of_comp (hg : SurjectiveOnStalks (g.comp f)) :
    SurjectiveOnStalks g := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    T : Type u_3
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : (g.comp f).SurjectiveOnStalks
    ⊢ g.SurjectiveOnStalks
  -/
  intros I hI
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    T : Type u_3
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : (g.comp f).SurjectiveOnStalks
    I : Ideal T
    hI : I.IsPrime
    ⊢ Function.Surjective ⇑(Localization.localRingHom (Ideal.comap g I) I g ⋯)
  -/
  have := hg I hI
  rw [Localization.localRingHom_comp (I.comap (g.comp f)) (I.comap g) _ _ rfl _ rfl,
    RingHom.coe_comp] at this
  /-
    R : Type u_1
    inst✝² : CommRing R
    S : Type u_2
    inst✝¹ : CommRing S
    T : Type u_3
    inst✝ : CommRing T
    g : RingHom S T
    f : RingHom R S
    hg : (g.comp f).SurjectiveOnStalks
    I : Ideal T
    hI : I.IsPrime
    this : Function.Surjective (Function.comp ⇑(Localization.localRingHom (Ideal.c …
    ⊢ Function.Surjective ⇑(Localization.localRingHom (Ideal.comap g I) I g ⋯)
  -/
  exact this.of_comp
  /-
    🎉 no goals
  -/



variable [Algebra R T] [Algebra R S] in
/--
If `R → T` is surjective on stalks, and `J` is some prime of `T`,
then every element `x` in `S ⊗[R] T` satisfies `(1 ⊗ r • t) * x = a ⊗ t` for some
`r : R`, `a : S`, and `t : T` such that `r • t ∉ J`.
-/
lemma SurjectiveOnStalks.exists_mul_eq_tmul
    (hf₂ : (algebraMap R T).SurjectiveOnStalks)
    (x : S ⊗[R] T) (J : Ideal T) (hJ : J.IsPrime) :
    ∃ (t : T) (r : R) (a : S), (r • t ∉ J) ∧
      (1 : S) ⊗ₜ[R] (r • t) * x = a ⊗ₜ[R] t := by
  induction x with
  | zero =>
    exact ⟨1, 1, 0, by rw [one_smul]; exact J.primeCompl.one_mem,
      by rw [mul_zero, TensorProduct.zero_tmul]⟩
  | tmul x₁ x₂ =>
    obtain ⟨y, s, c, hs, hc, e⟩ := (surjective_localRingHom_iff _).mp (hf₂ J hJ) x₂
    simp_rw [Algebra.smul_def]
    refine ⟨c, s, y • x₁, J.primeCompl.mul_mem hc hs, ?_⟩
    rw [Algebra.TensorProduct.tmul_mul_tmul, one_mul, mul_comm _ c, e,
      TensorProduct.smul_tmul, Algebra.smul_def, mul_comm]
  | add x₁ x₂ hx₁ hx₂ =>
    obtain ⟨t₁, r₁, a₁, hr₁, e₁⟩ := hx₁
    obtain ⟨t₂, r₂, a₂, hr₂, e₂⟩ := hx₂
    have : (r₁ * r₂) • (t₁ * t₂) = (r₁ • t₁) * (r₂ • t₂) := by
      simp_rw [← smul_eq_mul]; rw [smul_smul_smul_comm]
    refine ⟨t₁ * t₂, r₁ * r₂, r₂ • a₁ + r₁ • a₂, this.symm ▸ J.primeCompl.mul_mem hr₁ hr₂, ?_⟩
    rw [this, ← one_mul (1 : S), ← Algebra.TensorProduct.tmul_mul_tmul, mul_add, mul_comm (_ ⊗ₜ _),
      mul_assoc, e₁, Algebra.TensorProduct.tmul_mul_tmul, one_mul, smul_mul_assoc,
      ← TensorProduct.smul_tmul, mul_comm (_ ⊗ₜ _), mul_assoc, e₂,
      Algebra.TensorProduct.tmul_mul_tmul, one_mul, smul_mul_assoc, ← TensorProduct.smul_tmul,
      TensorProduct.add_tmul, mul_comm t₁ t₂]


variable (S) in
lemma surjectiveOnStalks_of_isLocalization
    [Algebra R S] [IsLocalization M S] :
    SurjectiveOnStalks (algebraMap R S) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    ⊢ (algebraMap R S).SurjectiveOnStalks
  -/
  refine surjectiveOnStalks_of_exists_div fun s ↦ ?_
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    s : S
    ⊢ Exists fun r => Exists fun s_1 => And (IsUnit ((algebraMap R S) s_1)) (Eq (H …
  -/
  obtain ⟨x, s, rfl⟩ := IsLocalization.mk'_surjective M s
  /-
    case intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    M : Submonoid R
    S : Type u_2
    inst✝² : CommRing S
    inst✝¹ : Algebra R S
    inst✝ : IsLocalization M S
    x : R
    s : Subtype fun x => Membership.mem M x
    ⊢ Exists fun r => Exists fun s_1 => And (IsUnit ((algebraMap R S) s_1)) (Eq (H …
  -/
  exact ⟨x, s, IsLocalization.map_units S s, IsLocalization.mk'_spec' S x s⟩
  /-
    🎉 no goals
  -/


lemma SurjectiveOnStalks.baseChange
    [Algebra R T] [Algebra R S]
    (hf : (algebraMap R T).SurjectiveOnStalks) :
    (algebraMap S (S ⊗[R] T)).SurjectiveOnStalks := by
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    T : Type u_3
    inst✝² : CommRing T
    inst✝¹ : Algebra R T
    inst✝ : Algebra R S
    hf : (algebraMap R T).SurjectiveOnStalks
    ⊢ (algebraMap S (TensorProduct R S T)).SurjectiveOnStalks
  -/
  let g : T →+* S ⊗[R] T := Algebra.TensorProduct.includeRight.toRingHom
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    T : Type u_3
    inst✝² : CommRing T
    inst✝¹ : Algebra R T
    inst✝ : Algebra R S
    hf : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    ⊢ (algebraMap S (TensorProduct R S T)).SurjectiveOnStalks
  -/
  intros J hJ
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    T : Type u_3
    inst✝² : CommRing T
    inst✝¹ : Algebra R T
    inst✝ : Algebra R S
    hf : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    J : Ideal (TensorProduct R S T)
    hJ : J.IsPrime
    ⊢ Function.Surjective ⇑(Localization.localRingHom (Ideal.comap (algebraMap S ( …
  -/
  rw [surjective_localRingHom_iff]
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    T : Type u_3
    inst✝² : CommRing T
    inst✝¹ : Algebra R T
    inst✝ : Algebra R S
    hf : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    J : Ideal (TensorProduct R S T)
    hJ : J.IsPrime
    ⊢ ∀ (s : TensorProduct R S T), Exists fun x => Exists fun r => Exists fun c => …
  -/
  intro x
  /-
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    T : Type u_3
    inst✝² : CommRing T
    inst✝¹ : Algebra R T
    inst✝ : Algebra R S
    hf : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    J : Ideal (TensorProduct R S T)
    hJ : J.IsPrime
    x : TensorProduct R S T
    ⊢ Exists fun x_1 => Exists fun r => Exists fun c => And (Not (Membership.mem J …
  -/
  obtain ⟨t, r, a, ht, e⟩ := hf.exists_mul_eq_tmul x (J.comap g) inferInstance
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝⁴ : CommRing R
    S : Type u_2
    inst✝³ : CommRing S
    T : Type u_3
    inst✝² : CommRing T
    inst✝¹ : Algebra R T
    inst✝ : Algebra R S
    hf : (algebraMap R T).SurjectiveOnStalks
    g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
    J : Ideal (TensorProduct R S T)
    hJ : J.IsPrime
    x : TensorProduct R S T
    t : T
    r : R
    a : S
    ht : Not (Membership.mem (Ideal.comap g J) (HSMul.hSMul r t))
    e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
    ⊢ Exists fun x_1 => Exists fun r => Exists fun c => And (Not (Membership.mem J …
  -/
  refine ⟨a, algebraMap _ _ r, 1 ⊗ₜ (r • t), ht, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      T : Type u_3
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra R S
      hf : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      J : Ideal (TensorProduct R S T)
      hJ : J.IsPrime
      x : TensorProduct R S T
      t : T
      r : R
      a : S
      ht : Not (Membership.mem (Ideal.comap g J) (HSMul.hSMul r t))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
      ⊢ Not (Membership.mem J ((algebraMap S (TensorProduct R S T)) ((algebraMap R S …
    -/
  · intro H
    simp only [Algebra.algebraMap_eq_smul_one (A := S), Algebra.TensorProduct.algebraMap_apply,
      Algebra.id.map_eq_id, id_apply, smul_tmul, ← Algebra.algebraMap_eq_smul_one (A := T)] at H
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      T : Type u_3
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra R S
      hf : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      J : Ideal (TensorProduct R S T)
      hJ : J.IsPrime
      x : TensorProduct R S T
      t : T
      r : R
      a : S
      ht : Not (Membership.mem (Ideal.comap g J) (HSMul.hSMul r t))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
      H : Membership.mem J (TensorProduct.tmul R 1 ((algebraMap R T) r))
      ⊢ False
    -/
    rw [Ideal.mem_comap, Algebra.smul_def, g.map_mul] at ht
    /-
      case intro.intro.intro.intro.refine_1
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      T : Type u_3
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra R S
      hf : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      J : Ideal (TensorProduct R S T)
      hJ : J.IsPrime
      x : TensorProduct R S T
      t : T
      r : R
      a : S
      ht : Not (Membership.mem J (HMul.hMul (g ((algebraMap R T) r)) (g t)))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
      H : Membership.mem J (TensorProduct.tmul R 1 ((algebraMap R T) r))
      ⊢ False
    -/
    exact ht (J.mul_mem_right _ H)
    /-
      🎉 no goals
    -/
  · simp only [tmul_smul, Algebra.TensorProduct.algebraMap_apply, Algebra.id.map_eq_id,
      RingHomCompTriple.comp_apply, Algebra.smul_mul_assoc, Algebra.TensorProduct.tmul_mul_tmul,
      one_mul, mul_one, id_apply, ← e]
    /-
      case intro.intro.intro.intro.refine_2
      R : Type u_1
      inst✝⁴ : CommRing R
      S : Type u_2
      inst✝³ : CommRing S
      T : Type u_3
      inst✝² : CommRing T
      inst✝¹ : Algebra R T
      inst✝ : Algebra R S
      hf : (algebraMap R T).SurjectiveOnStalks
      g : RingHom T (TensorProduct R S T) := Algebra.TensorProduct.includeRight.toRi …
      J : Ideal (TensorProduct R S T)
      hJ : J.IsPrime
      x : TensorProduct R S T
      t : T
      r : R
      a : S
      ht : Not (Membership.mem (Ideal.comap g J) (HSMul.hSMul r t))
      e : Eq (HMul.hMul (TensorProduct.tmul R 1 (HSMul.hSMul r t)) x) (TensorProduct …
      ⊢ Eq (HSMul.hSMul r (HMul.hMul (TensorProduct.tmul R ((algebraMap R S) r) t) x …
    -/
    rw [Algebra.algebraMap_eq_smul_one, ← smul_tmul', smul_mul_assoc]
    /-
      🎉 no goals
    -/


lemma surjectiveOnStalks_iff_of_isLocalHom [IsLocalRing S] [IsLocalHom f] :
    f.SurjectiveOnStalks ↔ Function.Surjective f := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    f : RingHom R S
    inst✝¹ : IsLocalRing S
    inst✝ : IsLocalHom f
    ⊢ Iff f.SurjectiveOnStalks (Function.Surjective ⇑f)
  -/
  refine ⟨fun H x ↦ ?_, fun h ↦ surjectiveOnStalks_of_surjective h⟩
  obtain ⟨y, r, c, hc, hr, e⟩ :=
    (surjective_localRingHom_iff _).mp (H (IsLocalRing.maximalIdeal _) inferInstance) x
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    f : RingHom R S
    inst✝¹ : IsLocalRing S
    inst✝ : IsLocalHom f
    H : f.SurjectiveOnStalks
    x : S
    y r : R
    c : S
    hc : Not (Membership.mem (IsLocalRing.maximalIdeal S) c)
    hr : Not (Membership.mem (IsLocalRing.maximalIdeal S) (f r))
    e : Eq (HMul.hMul (HMul.hMul c (f r)) x) (HMul.hMul c (f y))
    ⊢ Exists fun a => Eq (f a) x
  -/
  simp only [IsLocalRing.mem_maximalIdeal, mem_nonunits_iff, not_not] at hc hr
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    f : RingHom R S
    inst✝¹ : IsLocalRing S
    inst✝ : IsLocalHom f
    H : f.SurjectiveOnStalks
    x : S
    y r : R
    c : S
    e : Eq (HMul.hMul (HMul.hMul c (f r)) x) (HMul.hMul c (f y))
    hc : IsUnit c
    hr : IsUnit (f r)
    ⊢ Exists fun a => Eq (f a) x
  -/
  refine ⟨(isUnit_of_map_unit f r hr).unit⁻¹ * y, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    f : RingHom R S
    inst✝¹ : IsLocalRing S
    inst✝ : IsLocalHom f
    H : f.SurjectiveOnStalks
    x : S
    y r : R
    c : S
    e : Eq (HMul.hMul (HMul.hMul c (f r)) x) (HMul.hMul c (f y))
    hc : IsUnit c
    hr : IsUnit (f r)
    ⊢ Eq (f (HMul.hMul (↑(Inv.inv ⋯.unit)) y)) x
  -/
  apply hr.mul_right_injective
  /-
    case intro.intro.intro.intro.intro.a
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    f : RingHom R S
    inst✝¹ : IsLocalRing S
    inst✝ : IsLocalHom f
    H : f.SurjectiveOnStalks
    x : S
    y r : R
    c : S
    e : Eq (HMul.hMul (HMul.hMul c (f r)) x) (HMul.hMul c (f y))
    hc : IsUnit c
    hr : IsUnit (f r)
    ⊢ Eq ((fun x => HMul.hMul (f r) x) (f (HMul.hMul (↑(Inv.inv ⋯.unit)) y))) ((fu …
  -/
  apply hc.mul_right_injective
  /-
    case intro.intro.intro.intro.intro.a.a
    R : Type u_1
    inst✝³ : CommRing R
    S : Type u_2
    inst✝² : CommRing S
    f : RingHom R S
    inst✝¹ : IsLocalRing S
    inst✝ : IsLocalHom f
    H : f.SurjectiveOnStalks
    x : S
    y r : R
    c : S
    e : Eq (HMul.hMul (HMul.hMul c (f r)) x) (HMul.hMul c (f y))
    hc : IsUnit c
    hr : IsUnit (f r)
    ⊢ Eq ((fun x => HMul.hMul c x) ((fun x => HMul.hMul (f r) x) (f (HMul.hMul (↑( …
  -/
  simp only [← _root_.map_mul, ← mul_assoc, IsUnit.mul_val_inv, one_mul, e]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-10")]
alias surjectiveOnStalks_iff_of_isLocalRingHom := surjectiveOnStalks_iff_of_isLocalHom


