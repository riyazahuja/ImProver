theorem MulAction.smul_bijective_of_is_unit
    {M : Type*} [Monoid M] {α : Type*} [MulAction M α] {m : M} (hm : IsUnit m) :
    Function.Bijective (fun (a : α) ↦ m • a) := by
  /-
    M : Type u_1
    inst✝¹ : Monoid M
    α : Type u_2
    inst✝ : MulAction M α
    m : M
    hm : IsUnit m
    ⊢ Function.Bijective fun a => HSMul.hSMul m a
  -/
  lift m to Mˣ using hm
  /-
    case intro
    M : Type u_1
    inst✝¹ : Monoid M
    α : Type u_2
    inst✝ : MulAction M α
    m : Units M
    ⊢ Function.Bijective fun a => HSMul.hSMul (↑m) a
  -/
  rw [Function.bijective_iff_has_inverse]
  /-
    case intro
    M : Type u_1
    inst✝¹ : Monoid M
    α : Type u_2
    inst✝ : MulAction M α
    m : Units M
    ⊢ Exists fun g => And (Function.LeftInverse g fun a => HSMul.hSMul (↑m) a) (Fu …
  -/
  use fun a ↦ m⁻¹ • a
  /-
    case h
    M : Type u_1
    inst✝¹ : Monoid M
    α : Type u_2
    inst✝ : MulAction M α
    m : Units M
    ⊢ And (Function.LeftInverse (fun a => HSMul.hSMul (Inv.inv m) a) fun a => HSMu …
  -/
  constructor
    /-
      case h.left
      M : Type u_1
      inst✝¹ : Monoid M
      α : Type u_2
      inst✝ : MulAction M α
      m : Units M
      ⊢ Function.LeftInverse (fun a => HSMul.hSMul (Inv.inv m) a) fun a => HSMul.hSM …
    -/
  · intro x; simp [← Units.smul_def]
             /-
               🎉 no goals
             -/
    /-
      case h.right
      M : Type u_1
      inst✝¹ : Monoid M
      α : Type u_2
      inst✝ : MulAction M α
      m : Units M
      ⊢ Function.RightInverse (fun a => HSMul.hSMul (Inv.inv m) a) fun a => HSMul.hS …
    -/
  · intro x; simp [← Units.smul_def]
             /-
               🎉 no goals
             -/


theorem image_smul_setₛₗ :
    h '' (c • s) = σ c • h '' s := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_6
    inst✝⁵ : Monoid R
    inst✝⁴ : Monoid S
    σ : R → S
    inst✝³ : MulAction R M
    inst✝² : MulAction S N
    F : Type u_7
    h : F
    inst✝¹ : FunLike F M N
    inst✝ : MulActionSemiHomClass F σ M N
    c : R
    s : Set M
    ⊢ Eq (Set.image (⇑h) (HSMul.hSMul c s)) (HSMul.hSMul (σ c) (Set.image (⇑h) s))
  -/
  simp only [← image_smul, image_image, map_smulₛₗ h]
  /-
    🎉 no goals
  -/


/-- Translation of preimage is contained in preimage of translation -/
theorem smul_preimage_set_leₛₗ :
    c • h ⁻¹' t ⊆ h ⁻¹' (σ c • t) := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_6
    inst✝⁵ : Monoid R
    inst✝⁴ : Monoid S
    σ : R → S
    inst✝³ : MulAction R M
    inst✝² : MulAction S N
    F : Type u_7
    h : F
    inst✝¹ : FunLike F M N
    inst✝ : MulActionSemiHomClass F σ M N
    c : R
    t : Set N
    ⊢ HasSubset.Subset (HSMul.hSMul c (Set.preimage (⇑h) t)) (Set.preimage (⇑h) (H …
  -/
  rintro x ⟨y, hy, rfl⟩
  /-
    case intro.intro
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_6
    inst✝⁵ : Monoid R
    inst✝⁴ : Monoid S
    σ : R → S
    inst✝³ : MulAction R M
    inst✝² : MulAction S N
    F : Type u_7
    h : F
    inst✝¹ : FunLike F M N
    inst✝ : MulActionSemiHomClass F σ M N
    c : R
    t : Set N
    y : M
    hy : Membership.mem (Set.preimage (⇑h) t) y
    ⊢ Membership.mem (Set.preimage (⇑h) (HSMul.hSMul (σ c) t)) ((fun x => HSMul.hS …
  -/
  exact ⟨h y, hy, by rw [map_smulₛₗ]⟩
  /-
    🎉 no goals
  -/


/-- General version of `preimage_smul_setₛₗ` -/
theorem preimage_smul_setₛₗ'
    (hc : Function.Surjective (fun (m : M) ↦ c • m))
    (hc' : Function.Injective (fun (n : N) ↦ σ c • n)) :
    h ⁻¹' (σ c • t) = c • h ⁻¹' t := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_6
    inst✝⁵ : Monoid R
    inst✝⁴ : Monoid S
    σ : R → S
    inst✝³ : MulAction R M
    inst✝² : MulAction S N
    F : Type u_7
    h : F
    inst✝¹ : FunLike F M N
    inst✝ : MulActionSemiHomClass F σ M N
    c : R
    t : Set N
    hc : Function.Surjective fun m => HSMul.hSMul c m
    hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
    ⊢ Eq (Set.preimage (⇑h) (HSMul.hSMul (σ c) t)) (HSMul.hSMul c (Set.preimage (⇑ …
  -/
  apply le_antisymm
    /-
      case a
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      ⊢ LE.le (Set.preimage (⇑h) (HSMul.hSMul (σ c) t)) (HSMul.hSMul c (Set.preimage …
    -/
  · intro m
    /-
      case a
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      m : M
      ⊢ Membership.mem (Set.preimage (⇑h) (HSMul.hSMul (σ c) t)) m → Membership.mem  …
    -/
    obtain ⟨m', rfl⟩ := hc m
    /-
      case a.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      m' : M
      ⊢ Membership.mem (Set.preimage (⇑h) (HSMul.hSMul (σ c) t)) ((fun m => HSMul.hS …
    -/
    rintro ⟨n, hn, hn'⟩
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      m' : M
      n : N
      hn : Membership.mem t n
      hn' : Eq ((fun x => HSMul.hSMul (σ c) x) n) (h ((fun m => HSMul.hSMul c m) m'))
      ⊢ Membership.mem (HSMul.hSMul c (Set.preimage (⇑h) t)) ((fun m => HSMul.hSMul  …
    -/
    refine ⟨m', ?_, rfl⟩
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      m' : M
      n : N
      hn : Membership.mem t n
      hn' : Eq ((fun x => HSMul.hSMul (σ c) x) n) (h ((fun m => HSMul.hSMul c m) m'))
      ⊢ Membership.mem (Set.preimage (⇑h) t) m'
    -/
    rw [map_smulₛₗ] at hn'
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      m' : M
      n : N
      hn : Membership.mem t n
      hn' : Eq ((fun x => HSMul.hSMul (σ c) x) n) (HSMul.hSMul (σ c) (h m'))
      ⊢ Membership.mem (Set.preimage (⇑h) t) m'
    -/
    rw [mem_preimage, ← hc' hn']
    /-
      case a.intro.intro.intro
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      m' : M
      n : N
      hn : Membership.mem t n
      hn' : Eq ((fun x => HSMul.hSMul (σ c) x) n) (HSMul.hSMul (σ c) (h m'))
      ⊢ Membership.mem t n
    -/
    exact hn
    /-
      🎉 no goals
    -/
    /-
      case a
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : Function.Surjective fun m => HSMul.hSMul c m
      hc' : Function.Injective fun n => HSMul.hSMul (σ c) n
      ⊢ LE.le (HSMul.hSMul c (Set.preimage (⇑h) t)) (Set.preimage (⇑h) (HSMul.hSMul  …
    -/
  · exact smul_preimage_set_leₛₗ M N σ h c t
    /-
      🎉 no goals
    -/


/-- `preimage_smul_setₛₗ` when both scalars act by unit -/
theorem preimage_smul_setₛₗ_of_units (hc : IsUnit c) (hc' : IsUnit (σ c)) :
    h ⁻¹' (σ c • t) = c • h ⁻¹' t := by
  /-
    R : Type u_1
    S : Type u_2
    M : Type u_3
    N : Type u_6
    inst✝⁵ : Monoid R
    inst✝⁴ : Monoid S
    σ : R → S
    inst✝³ : MulAction R M
    inst✝² : MulAction S N
    F : Type u_7
    h : F
    inst✝¹ : FunLike F M N
    inst✝ : MulActionSemiHomClass F σ M N
    c : R
    t : Set N
    hc : IsUnit c
    hc' : IsUnit (σ c)
    ⊢ Eq (Set.preimage (⇑h) (HSMul.hSMul (σ c) t)) (HSMul.hSMul c (Set.preimage (⇑ …
  -/
  apply preimage_smul_setₛₗ'
    /-
      case hc
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : IsUnit c
      hc' : IsUnit (σ c)
      ⊢ Function.Surjective fun m => HSMul.hSMul c m
    -/
  · exact (MulAction.smul_bijective_of_is_unit hc).surjective
    /-
      🎉 no goals
    -/
    /-
      case hc'
      R : Type u_1
      S : Type u_2
      M : Type u_3
      N : Type u_6
      inst✝⁵ : Monoid R
      inst✝⁴ : Monoid S
      σ : R → S
      inst✝³ : MulAction R M
      inst✝² : MulAction S N
      F : Type u_7
      h : F
      inst✝¹ : FunLike F M N
      inst✝ : MulActionSemiHomClass F σ M N
      c : R
      t : Set N
      hc : IsUnit c
      hc' : IsUnit (σ c)
      ⊢ Function.Injective fun n => HSMul.hSMul (σ c) n
    -/
  · exact (MulAction.smul_bijective_of_is_unit hc').injective
    /-
      🎉 no goals
    -/



/-- `preimage_smul_setₛₗ` in the context of a `MonoidHom` -/
theorem MonoidHom.preimage_smul_setₛₗ (σ : R →* S)
    {F : Type*} [FunLike F M N] [MulActionSemiHomClass F ⇑σ M N] (h : F)
    {c : R} (hc : IsUnit c) (t : Set N) :
    h ⁻¹' (σ c • t) = c • h ⁻¹' t :=
  preimage_smul_setₛₗ_of_units M N σ h t hc (IsUnit.map σ hc)


/-- `preimage_smul_setₛₗ` in the context of a `MonoidHomClass` -/
theorem preimage_smul_setₛₗ
    {G : Type*} [FunLike G R S] [MonoidHomClass G R S] (σ : G)
    {F : Type*} [FunLike F M N] [MulActionSemiHomClass F σ M N] (h : F)
    {c : R} (hc : IsUnit c) (t : Set N) :
    h ⁻¹' (σ c • t) = c • h ⁻¹' t :=
 MonoidHom.preimage_smul_setₛₗ M N (σ : R →* S) h hc t


/-- `preimage_smul_setₛₗ` in the context of a groups -/
theorem Group.preimage_smul_setₛₗ
    {R S : Type*} [Group R] [Group S] (σ : R → S)
    [MulAction R M] [MulAction S N]
    {F : Type*} [FunLike F M N] [MulActionSemiHomClass F σ M N] (h : F)
    (c : R) (t : Set N) :
    h ⁻¹' (σ c • t) = c • h ⁻¹' t :=
  preimage_smul_setₛₗ_of_units M N σ h t (Group.isUnit _) (Group.isUnit _)


@[simp] -- This can be safely removed as a `@[simp]` lemma if `image_smul_setₛₗ` is readded.
theorem image_smul_set :
    h '' (c • s) = c • h '' s :=
  image_smul_setₛₗ _ _ _ h c s


theorem smul_preimage_set_le :
    c • h ⁻¹' t ⊆ h ⁻¹' (c • t) :=
  smul_preimage_set_leₛₗ _ _ _ h c t


theorem preimage_smul_set (hc : IsUnit c) :
    h ⁻¹' (c • t) = c • h ⁻¹' t :=
  preimage_smul_setₛₗ_of_units _ _ _ h t hc hc


theorem Group.preimage_smul_set
    {R : Type*} [Group R] (M₁ M₂ : Type*)
    [MulAction R M₁] [MulAction R M₂]
    {F : Type*} [FunLike F M₁ M₂] [MulActionHomClass F R M₁ M₂] (h : F)
    (c : R) (t : Set M₂) :
    h ⁻¹' (c • t) = c • h ⁻¹' t :=
  _root_.preimage_smul_set R M₁ M₂ h t (Group.isUnit c)


