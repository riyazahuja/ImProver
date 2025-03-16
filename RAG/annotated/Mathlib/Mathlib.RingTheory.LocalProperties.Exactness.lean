theorem injective_of_isLocalized_maximal
    (H : ∀ (P : Ideal R) [P.IsMaximal], Function.Injective (map P.primeCompl (f P) (g P) F)) :
    Function.Injective F :=
                                                                             /-
                                                                               R : Type u_1
                                                                               M : Type u_2
                                                                               N : Type u_3
                                                                               inst✝¹⁰ : CommSemiring R
                                                                               inst✝⁹ : AddCommMonoid M
                                                                               inst✝⁸ : Module R M
                                                                               inst✝⁷ : AddCommMonoid N
                                                                               inst✝⁶ : Module R N
                                                                               Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
                                                                               inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
                                                                               inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
                                                                               f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
                                                                               inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
                                                                               Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
                                                                               inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
                                                                               inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
                                                                               g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
                                                                               inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
                                                                               F : LinearMap (RingHom.id R) M N
                                                                               H : ∀ (P : Ideal R) [inst : P.IsMaximal], Function.Injective ⇑((IsLocalizedMod …
                                                                               x y : M
                                                                               eq : Eq (F x) (F y)
                                                                               P : Ideal R
                                                                               x✝ : P.IsMaximal
                                                                               ⊢ Eq (((IsLocalizedModule.map P.primeCompl (f P) (g P)) F) ((f P) x)) (((IsLoc …
                                                                             -/
  fun x y eq ↦ Module.eq_of_localization_maximal _ f _ _ fun P _ ↦ H P <| by simp [eq]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem surjective_of_isLocalized_maximal
    (H : ∀ (P : Ideal R) [P.IsMaximal], Function.Surjective (map P.primeCompl (f P) (g P) F)) :
    Function.Surjective F :=
  range_eq_top.mp <| eq_top_of_localization₀_maximal Nₚ g _ <|
    fun P _ ↦ (range_localizedMap_eq_localized₀_range _ (f P) (g P) F).symm.trans <|
    range_eq_top.mpr <| H P


theorem bijective_of_isLocalized_maximal
    (H : ∀ (P : Ideal R) [P.IsMaximal], Function.Bijective (map P.primeCompl (f P) (g P) F)) :
    Function.Bijective F :=
  ⟨injective_of_isLocalized_maximal Mₚ f Nₚ g F fun J _ ↦ (H J).1,
  surjective_of_isLocalized_maximal Mₚ f Nₚ g F fun J _ ↦ (H J).2⟩


theorem exact_of_isLocalized_maximal (H : ∀ (J : Ideal R) [J.IsMaximal],
    Function.Exact (map J.primeCompl (f J) (g J) F) (map J.primeCompl (g J) (h J) G)) :
    Function.Exact F G := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁸ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝⁶ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
    inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
    g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
    inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Lₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_7
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Lₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Lₚ P)
    h : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) L (Lₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (J : Ideal R) [inst : J.IsMaximal], Function.Exact ⇑((IsLocalizedModule. …
    ⊢ Function.Exact ⇑F ⇑G
  -/
  simp only [LinearMap.exact_iff] at H ⊢
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁸ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝⁶ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
    inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
    g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
    inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Lₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_7
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Lₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Lₚ P)
    h : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) L (Lₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (J : Ideal R) [inst : J.IsMaximal], Eq (LinearMap.ker ((IsLocalizedModul …
    ⊢ Eq (LinearMap.ker G) (LinearMap.range F)
  -/
  apply eq_of_localization₀_maximal Nₚ g
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁸ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝⁶ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
    inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
    g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
    inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Lₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_7
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Lₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Lₚ P)
    h : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) L (Lₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (J : Ideal R) [inst : J.IsMaximal], Eq (LinearMap.ker ((IsLocalizedModul …
    ⊢ ∀ (P : Ideal R) [inst : P.IsMaximal], Eq (Submodule.localized₀ P.primeCompl  …
  -/
  intro J hJ
  rw [← LinearMap.range_localizedMap_eq_localized₀_range _ (f J) (g J) F,
    ← LinearMap.ker_localizedMap_eq_localized₀_ker J.primeCompl (g J) (h J) G]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁸ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝⁶ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
    inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
    g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
    inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Lₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_7
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Lₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Lₚ P)
    h : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) L (Lₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (J : Ideal R) [inst : J.IsMaximal], Eq (LinearMap.ker ((IsLocalizedModul …
    J : Ideal R
    hJ : J.IsMaximal
    ⊢ Eq (LinearMap.ker ((IsLocalizedModule.map J.primeCompl (g J) (h J)) G)) (Lin …
  -/
  have := SetLike.ext_iff.mp <| H J
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁸ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝⁶ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
    inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
    g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
    inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Lₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_7
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Lₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Lₚ P)
    h : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) L (Lₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (J : Ideal R) [inst : J.IsMaximal], Eq (LinearMap.ker ((IsLocalizedModul …
    J : Ideal R
    hJ : J.IsMaximal
    this : ∀ (x : Nₚ J), Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.ma …
    ⊢ Eq (LinearMap.ker ((IsLocalizedModule.map J.primeCompl (g J) (h J)) G)) (Lin …
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁸ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝⁶ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
    inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
    g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
    inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Lₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_7
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Lₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Lₚ P)
    h : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) L (Lₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (J : Ideal R) [inst : J.IsMaximal], Eq (LinearMap.ker ((IsLocalizedModul …
    J : Ideal R
    hJ : J.IsMaximal
    this : ∀ (x : Nₚ J), Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.ma …
    x : Nₚ J
    ⊢ Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.map J.primeCompl (g J …
  -/
  simp only [mem_range, mem_ker] at this ⊢
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    Mₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_5
    inst✝⁸ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Mₚ P)
    inst✝⁷ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Mₚ P)
    f : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) M (Mₚ P)
    inst✝⁶ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Nₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_6
    inst✝⁵ : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Nₚ P)
    inst✝⁴ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Nₚ P)
    g : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) N (Nₚ P)
    inst✝³ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl  …
    Lₚ : (P : Ideal R) → [inst : P.IsMaximal] → Type u_7
    inst✝² : (P : Ideal R) → [inst : P.IsMaximal] → AddCommMonoid (Lₚ P)
    inst✝¹ : (P : Ideal R) → [inst : P.IsMaximal] → Module R (Lₚ P)
    h : (P : Ideal R) → [inst : P.IsMaximal] → LinearMap (RingHom.id R) L (Lₚ P)
    inst✝ : ∀ (P : Ideal R) [inst : P.IsMaximal], IsLocalizedModule P.primeCompl ( …
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (J : Ideal R) [inst : J.IsMaximal], Eq (LinearMap.ker ((IsLocalizedModul …
    J : Ideal R
    hJ : J.IsMaximal
    x : Nₚ J
    this : ∀ (x : Nₚ J), Iff (Eq (((IsLocalizedModule.map J.primeCompl (g J) (h J) …
    ⊢ Iff (Eq (((IsLocalizedModule.map J.primeCompl (g J) (h J)) G) x) 0) (Exists  …
  -/
  exact this x
  /-
    🎉 no goals
  -/


theorem injective_of_localized_maximal
    (h : ∀ (J : Ideal R) [J.IsMaximal], Function.Injective (map J.primeCompl f)) :
    Function.Injective f :=
  injective_of_isLocalized_maximal _ (fun _ _ ↦ mkLinearMap _ _) _ (fun _ _ ↦ mkLinearMap _ _) f h


theorem surjective_of_localized_maximal
    (h : ∀ (J : Ideal R) [J.IsMaximal], Function.Surjective (map J.primeCompl f)) :
    Function.Surjective f :=
  surjective_of_isLocalized_maximal _ (fun _ _ ↦ mkLinearMap _ _) _ (fun _ _ ↦ mkLinearMap _ _) f h


theorem bijective_of_localized_maximal
    (h : ∀ (J : Ideal R) [J.IsMaximal], Function.Bijective (map J.primeCompl f)) :
    Function.Bijective f :=
  ⟨injective_of_localized_maximal _ fun J _ ↦ (h J).1,
  surjective_of_localized_maximal _ fun J _ ↦ (h J).2⟩


theorem exact_of_localized_maximal
    (h : ∀ (J : Ideal R) [J.IsMaximal], Function.Exact (map J.primeCompl f) (map J.primeCompl g)) :
    Function.Exact f g :=
  exact_of_isLocalized_maximal _ (fun _ _ ↦ mkLinearMap _ _) _ (fun _ _ ↦ mkLinearMap _ _)
    _ (fun _ _ ↦ mkLinearMap _ _) f g h


theorem injective_of_isLocalized_span
    (H : ∀ r : s, Function.Injective (map (.powers r.1) (f r) (g r) F)) :
    Function.Injective F :=
                                                                             /-
                                                                               R : Type u_1
                                                                               M : Type u_2
                                                                               N : Type u_3
                                                                               inst✝¹⁰ : CommSemiring R
                                                                               inst✝⁹ : AddCommMonoid M
                                                                               inst✝⁸ : Module R M
                                                                               inst✝⁷ : AddCommMonoid N
                                                                               inst✝⁶ : Module R N
                                                                               s : Set R
                                                                               spn : Eq (Ideal.span s) Top.top
                                                                               Mₚ : ↑s → Type u_5
                                                                               inst✝⁵ : (r : ↑s) → AddCommMonoid (Mₚ r)
                                                                               inst✝⁴ : (r : ↑s) → Module R (Mₚ r)
                                                                               f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
                                                                               inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
                                                                               Nₚ : ↑s → Type u_6
                                                                               inst✝² : (r : ↑s) → AddCommMonoid (Nₚ r)
                                                                               inst✝¹ : (r : ↑s) → Module R (Nₚ r)
                                                                               g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
                                                                               inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
                                                                               F : LinearMap (RingHom.id R) M N
                                                                               H : ∀ (r : ↑s), Function.Injective ⇑((IsLocalizedModule.map (Submonoid.powers  …
                                                                               x y : M
                                                                               eq : Eq (F x) (F y)
                                                                               P : ↑s
                                                                               ⊢ Eq (((IsLocalizedModule.map (Submonoid.powers ↑P) (f P) (g P)) F) ((f P) x)) …
                                                                             -/
  fun x y eq ↦ Module.eq_of_isLocalized_span _ spn _ f _ _ fun P ↦ H P <| by simp [eq]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem surjective_of_isLocalized_span
    (H : ∀ r : s, Function.Surjective (map (.powers r.1) (f r) (g r) F)) :
    Function.Surjective F :=
  range_eq_top.mp <| eq_top_of_isLocalized₀_span s spn Nₚ g fun r ↦
    (range_localizedMap_eq_localized₀_range _ (f r) (g r) F).symm.trans <| range_eq_top.mpr <| H r


theorem bijective_of_isLocalized_span
    (H : ∀ r : s, Function.Bijective (map (.powers r.1) (f r) (g r) F)) :
    Function.Bijective F :=
  ⟨injective_of_isLocalized_span _ spn Mₚ f Nₚ g F fun r ↦ (H r).1,
  surjective_of_isLocalized_span _ spn Mₚ f Nₚ g F fun r ↦ (H r).2⟩


lemma exact_of_isLocalized_span (H : ∀ r : s, Function.Exact
    (map (.powers r.1) (f r) (g r) F) (map (.powers r.1) (g r) (h r) G)) :
    Function.Exact F G := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Function.Exact ⇑((IsLocalizedModule.map (Submonoid.powers ↑r)  …
    ⊢ Function.Exact ⇑F ⇑G
  -/
  simp only [LinearMap.exact_iff] at H ⊢
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    ⊢ Eq (LinearMap.ker G) (LinearMap.range F)
  -/
  apply Submodule.eq_of_isLocalized₀_span s spn Nₚ g
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    ⊢ ∀ (r : ↑s), Eq (Submodule.localized₀ (Submonoid.powers ↑r) (g r) (LinearMap. …
  -/
  intro r
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    r : ↑s
    ⊢ Eq (Submodule.localized₀ (Submonoid.powers ↑r) (g r) (LinearMap.ker G)) (Sub …
  -/
  rw [← LinearMap.range_localizedMap_eq_localized₀_range _ (f r) (g r) F]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    r : ↑s
    ⊢ Eq (Submodule.localized₀ (Submonoid.powers ↑r) (g r) (LinearMap.ker G)) (Lin …
  -/
  rw [← LinearMap.ker_localizedMap_eq_localized₀_ker (.powers r.1) (g r) (h r) G]
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    r : ↑s
    ⊢ Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r) (g r) (h r)) …
  -/
  have := SetLike.ext_iff.mp <| H r
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    r : ↑s
    this : ∀ (x : Nₚ r), Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.ma …
    ⊢ Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r) (g r) (h r)) …
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    r : ↑s
    this : ∀ (x : Nₚ r), Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.ma …
    x : Nₚ r
    ⊢ Iff (Membership.mem (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers …
  -/
  simp only [mem_range, mem_ker] at this ⊢
  /-
    case h
    R : Type u_1
    M : Type u_2
    N : Type u_3
    L : Type u_4
    inst✝¹⁵ : CommSemiring R
    inst✝¹⁴ : AddCommMonoid M
    inst✝¹³ : Module R M
    inst✝¹² : AddCommMonoid N
    inst✝¹¹ : Module R N
    inst✝¹⁰ : AddCommMonoid L
    inst✝⁹ : Module R L
    s : Set R
    spn : Eq (Ideal.span s) Top.top
    Mₚ : ↑s → Type u_5
    inst✝⁸ : (r : ↑s) → AddCommMonoid (Mₚ r)
    inst✝⁷ : (r : ↑s) → Module R (Mₚ r)
    f : (r : ↑s) → LinearMap (RingHom.id R) M (Mₚ r)
    inst✝⁶ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (f r)
    Nₚ : ↑s → Type u_6
    inst✝⁵ : (r : ↑s) → AddCommMonoid (Nₚ r)
    inst✝⁴ : (r : ↑s) → Module R (Nₚ r)
    g : (r : ↑s) → LinearMap (RingHom.id R) N (Nₚ r)
    inst✝³ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (g r)
    Lₚ : ↑s → Type u_7
    inst✝² : (r : ↑s) → AddCommMonoid (Lₚ r)
    inst✝¹ : (r : ↑s) → Module R (Lₚ r)
    h : (r : ↑s) → LinearMap (RingHom.id R) L (Lₚ r)
    inst✝ : ∀ (r : ↑s), IsLocalizedModule (Submonoid.powers ↑r) (h r)
    F : LinearMap (RingHom.id R) M N
    G : LinearMap (RingHom.id R) N L
    H : ∀ (r : ↑s), Eq (LinearMap.ker ((IsLocalizedModule.map (Submonoid.powers ↑r …
    r : ↑s
    x : Nₚ r
    this : ∀ (x : Nₚ r), Iff (Eq (((IsLocalizedModule.map (Submonoid.powers ↑r) (g …
    ⊢ Iff (Eq (((IsLocalizedModule.map (Submonoid.powers ↑r) (g r) (h r)) G) x) 0) …
  -/
  exact this x
  /-
    🎉 no goals
  -/


theorem injective_of_localized_span
    (h : ∀ r : s, Function.Injective (map (.powers r.1) f)) :
    Function.Injective f :=
  injective_of_isLocalized_span s spn _ (fun _ ↦ mkLinearMap _ _) _ (fun _ ↦ mkLinearMap _ _) f h


theorem surjective_of_localized_span
    (h : ∀ r : s, Function.Surjective (map (.powers r.1) f)) :
    Function.Surjective f :=
  surjective_of_isLocalized_span s spn _ (fun _ ↦ mkLinearMap _ _) _ (fun _ ↦ mkLinearMap _ _) f h


theorem bijective_of_localized_span
    (h : ∀ r : s, Function.Bijective (map (.powers r.1) f)) :
    Function.Bijective f :=
  ⟨injective_of_localized_span _ spn _ fun r ↦ (h r).1,
  surjective_of_localized_span _ spn _ fun r ↦ (h r).2⟩


lemma exact_of_localized_span
    (h : ∀ r : s, Function.Exact (map (.powers r.1) f) (map (.powers r.1) g)) :
    Function.Exact f g :=
  exact_of_isLocalized_span s spn _ (fun _ ↦ mkLinearMap _ _) _ (fun _ ↦ mkLinearMap _ _)
    _ (fun _ ↦ mkLinearMap _ _) f g h


