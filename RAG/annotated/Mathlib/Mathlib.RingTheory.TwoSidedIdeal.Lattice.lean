instance : SemilatticeSup (TwoSidedIdeal R) where
  sup I J := { ringCon := I.ringCon ⊔ J.ringCon }
                         /-
                           R : Type u_1
                           inst✝ : NonUnitalNonAssocRing R
                           I J : TwoSidedIdeal R
                           ⊢ LE.le I ((fun I J => { ringCon := Max.max I.ringCon J.ringCon }) I J)
                         -/
  le_sup_left I J :=  by rw [ringCon_le_iff]; exact le_sup_left
                                              /-
                                                🎉 no goals
                                              -/
                         /-
                           R : Type u_1
                           inst✝ : NonUnitalNonAssocRing R
                           I J : TwoSidedIdeal R
                           ⊢ LE.le J ((fun I J => { ringCon := Max.max I.ringCon J.ringCon }) I J)
                         -/
  le_sup_right I J := by rw [ringCon_le_iff]; exact le_sup_right
                                              /-
                                                🎉 no goals
                                              -/
                           /-
                             R : Type u_1
                             inst✝ : NonUnitalNonAssocRing R
                             I J K : TwoSidedIdeal R
                             h1 : LE.le I K
                             h2 : LE.le J K
                             ⊢ LE.le ((fun I J => { ringCon := Max.max I.ringCon J.ringCon }) I J) K
                           -/
  sup_le I J K h1 h2 := by rw [ringCon_le_iff] at h1 h2 ⊢; exact sup_le h1 h2
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma sup_ringCon (I J : TwoSidedIdeal R) : (I ⊔ J).ringCon = I.ringCon ⊔ J.ringCon := rfl


lemma mem_sup_left {I J : TwoSidedIdeal R} {x : R} (h : x ∈ I) :
    x ∈ I ⊔ J :=
  (show I ≤ I ⊔ J from le_sup_left) h


lemma mem_sup_right {I J : TwoSidedIdeal R} {x : R} (h : x ∈ J) :
    x ∈ I ⊔ J :=
  (show J ≤ I ⊔ J from le_sup_right) h


lemma mem_sup {I J : TwoSidedIdeal R} {x : R} :
    x ∈ I ⊔ J ↔ ∃ y ∈ I, ∃ z ∈ J, y + z = x := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    I J : TwoSidedIdeal R
    x : R
    ⊢ Iff (Membership.mem (Max.max I J) x) (Exists fun y => And (Membership.mem I  …
  -/
  constructor
  · let s : TwoSidedIdeal R := .mk'
      {x | ∃ y ∈ I, ∃ z ∈ J, y + z = x}
      ⟨0, ⟨zero_mem _, ⟨0, ⟨zero_mem _, zero_add _⟩⟩⟩⟩
      (by rintro _ _ ⟨x, ⟨hx, ⟨y, ⟨hy, rfl⟩⟩⟩⟩ ⟨a, ⟨ha, ⟨b, ⟨hb, rfl⟩⟩⟩⟩;
          exact ⟨x + a, ⟨add_mem _ hx ha, ⟨y + b, ⟨add_mem _ hy hb, by abel⟩⟩⟩⟩)
      (by rintro _ ⟨x, ⟨hx, ⟨y, ⟨hy, rfl⟩⟩⟩⟩
          exact ⟨-x, ⟨neg_mem _ hx, ⟨-y, ⟨neg_mem _ hy, by abel⟩⟩⟩⟩)
      (by rintro r _ ⟨x, ⟨hx, ⟨y, ⟨hy, rfl⟩⟩⟩⟩
          exact ⟨_, ⟨mul_mem_left _ _ _ hx, ⟨_, ⟨mul_mem_left _ _ _ hy, mul_add _ _ _ |>.symm⟩⟩⟩⟩)
      (by rintro r _ ⟨x, ⟨hx, ⟨y, ⟨hy, rfl⟩⟩⟩⟩
          exact ⟨_, ⟨mul_mem_right _ _ _ hx, ⟨_, ⟨mul_mem_right _ _ _ hy, add_mul _ _ _ |>.symm⟩⟩⟩⟩)
    suffices (I.ringCon ⊔ J.ringCon) ≤ s.ringCon by
      intro h; convert this h; rw [rel_iff, sub_zero, mem_mk']; rfl
    /-
      case mp
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I J : TwoSidedIdeal R
      x : R
      s : TwoSidedIdeal R := TwoSidedIdeal.mk' (setOf fun x => Exists fun y => And ( …
      ⊢ LE.le (Max.max I.ringCon J.ringCon) s.ringCon
    -/
    refine sup_le (fun x y h => ?_) (fun x y h => ?_) <;> rw [rel_iff] at h ⊢ <;> rw [mem_mk']
    /-
      case mp.refine_1
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I J : TwoSidedIdeal R
      x✝ : R
      s : TwoSidedIdeal R := TwoSidedIdeal.mk' (setOf fun x => Exists fun y => And ( …
      x y : R
      h : Membership.mem I (HSub.hSub x y)
      ⊢ Membership.mem (setOf fun x => Exists fun y => And (Membership.mem I y) (Exi …
    -/
    exacts [⟨_, ⟨h, ⟨0, ⟨zero_mem _, add_zero _⟩⟩⟩⟩, ⟨0, ⟨zero_mem _, ⟨_, ⟨h, zero_add _⟩⟩⟩⟩]
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u_1
      inst✝ : NonUnitalNonAssocRing R
      I J : TwoSidedIdeal R
      x : R
      ⊢ (Exists fun y => And (Membership.mem I y) (Exists fun z => And (Membership.m …
    -/
  · rintro ⟨y, ⟨hy, ⟨z, ⟨hz, rfl⟩⟩⟩⟩; exact add_mem _ (mem_sup_left hy) (mem_sup_right hz)
                                      /-
                                        🎉 no goals
                                      -/


instance : SemilatticeInf (TwoSidedIdeal R) where
  inf I J := { ringCon := I.ringCon ⊓ J.ringCon }
                        /-
                          R : Type u_1
                          inst✝ : NonUnitalNonAssocRing R
                          I J : TwoSidedIdeal R
                          ⊢ LE.le ((fun I J => { ringCon := Min.min I.ringCon J.ringCon }) I J) I
                        -/
  inf_le_left I J := by rw [ringCon_le_iff]; exact inf_le_left
                                             /-
                                               🎉 no goals
                                             -/
                         /-
                           R : Type u_1
                           inst✝ : NonUnitalNonAssocRing R
                           I J : TwoSidedIdeal R
                           ⊢ LE.le ((fun I J => { ringCon := Min.min I.ringCon J.ringCon }) I J) J
                         -/
  inf_le_right I J := by rw [ringCon_le_iff]; exact inf_le_right
                                              /-
                                                🎉 no goals
                                              -/
                           /-
                             R : Type u_1
                             inst✝ : NonUnitalNonAssocRing R
                             I J K : TwoSidedIdeal R
                             h1 : LE.le I J
                             h2 : LE.le I K
                             ⊢ LE.le I ((fun I J => { ringCon := Min.min I.ringCon J.ringCon }) J K)
                           -/
  le_inf I J K h1 h2 := by rw [ringCon_le_iff] at h1 h2 ⊢; exact le_inf h1 h2
                                                           /-
                                                             🎉 no goals
                                                           -/


lemma inf_ringCon (I J : TwoSidedIdeal R) : (I ⊓ J).ringCon = I.ringCon ⊓ J.ringCon := rfl


lemma mem_inf {I J : TwoSidedIdeal R} {x : R} :
    x ∈ I ⊓ J ↔ x ∈ I ∧ x ∈ J :=
  Iff.rfl


instance : SupSet (TwoSidedIdeal R) where
  sSup s := { ringCon := sSup <| TwoSidedIdeal.ringCon '' s }


lemma sSup_ringCon (S : Set (TwoSidedIdeal R)) :
    (sSup S).ringCon = sSup (TwoSidedIdeal.ringCon '' S) := rfl


lemma iSup_ringCon {ι : Type*} (I : ι → TwoSidedIdeal R) :
    (⨆ i, I i).ringCon = ⨆ i, (I i).ringCon := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    ι : Type u_2
    I : ι → TwoSidedIdeal R
    ⊢ Eq (iSup fun i => I i).ringCon (iSup fun i => (I i).ringCon)
  -/
  simp only [iSup, sSup_ringCon]; congr; ext; simp
                                              /-
                                                🎉 no goals
                                              -/


instance : CompleteSemilatticeSup (TwoSidedIdeal R) where
                      /-
                        R : Type u_1
                        inst✝ : NonUnitalNonAssocRing R
                        s : Set (TwoSidedIdeal R)
                        I : TwoSidedIdeal R
                        h : ∀ (b : TwoSidedIdeal R), Membership.mem s b → LE.le b I
                        ⊢ LE.le (SupSet.sSup s) I
                      -/
                       /-
                         R : Type u_1
                         inst✝ : NonUnitalNonAssocRing R
                         s : Set (TwoSidedIdeal R)
                         I : TwoSidedIdeal R
                         hI : Membership.mem s I
                         ⊢ LE.le I (SupSet.sSup s)
                       -/
  sSup_le s I h := by simp_rw [ringCon_le_iff] at h ⊢; exact sSup_le <| by aesop
                                            /-
                                              🎉 no goals
                                            -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  le_sSup s I hI := by rw [ringCon_le_iff]; exact le_sSup <| by aesop


instance : InfSet (TwoSidedIdeal R) where
  sInf s := { ringCon := sInf <| TwoSidedIdeal.ringCon '' s }


lemma sInf_ringCon (S : Set (TwoSidedIdeal R)) :
    (sInf S).ringCon = sInf (TwoSidedIdeal.ringCon '' S) := rfl


lemma iInf_ringCon {ι : Type*} (I : ι → TwoSidedIdeal R) :
    (⨅ i, I i).ringCon = ⨅ i, (I i).ringCon := by
  /-
    R : Type u_1
    inst✝ : NonUnitalNonAssocRing R
    ι : Type u_2
    I : ι → TwoSidedIdeal R
    ⊢ Eq (iInf fun i => I i).ringCon (iInf fun i => (I i).ringCon)
  -/
  simp only [iInf, sInf_ringCon]; congr!; ext; simp
                                               /-
                                                 🎉 no goals
                                               -/


instance : CompleteSemilatticeInf (TwoSidedIdeal R) where
                      /-
                        R : Type u_1
                        inst✝ : NonUnitalNonAssocRing R
                        s : Set (TwoSidedIdeal R)
                        I : TwoSidedIdeal R
                        h : ∀ (b : TwoSidedIdeal R), Membership.mem s b → LE.le I b
                        ⊢ LE.le I (InfSet.sInf s)
                      -/
                       /-
                         R : Type u_1
                         inst✝ : NonUnitalNonAssocRing R
                         s : Set (TwoSidedIdeal R)
                         I : TwoSidedIdeal R
                         hI : Membership.mem s I
                         ⊢ LE.le (InfSet.sInf s) I
                       -/
  le_sInf s I h := by simp_rw [ringCon_le_iff] at h ⊢; exact le_sInf <| by aesop
                                            /-
                                              🎉 no goals
                                            -/
                                                       /-
                                                         🎉 no goals
                                                       -/
  sInf_le s I hI := by rw [ringCon_le_iff]; exact sInf_le <| by aesop


lemma mem_iInf {ι : Type*} {I : ι → TwoSidedIdeal R} {x : R} :
    x ∈ iInf I ↔ ∀ i, x ∈ I i :=
                       /-
                         R : Type u_1
                         inst✝ : NonUnitalNonAssocRing R
                         ι : Type u_2
                         I : ι → TwoSidedIdeal R
                         x : R
                         ⊢ Iff (∀ (x_1 : RingCon R), Membership.mem (Set.image TwoSidedIdeal.ringCon (S …
                       -/
  show (∀ _, _) ↔ _ by simp [mem_iff]
                       /-
                         🎉 no goals
                       -/


lemma mem_sInf {S : Set (TwoSidedIdeal R)} {x : R} :
    x ∈ sInf S ↔ ∀ I ∈ S, x ∈ I :=
                       /-
                         R : Type u_1
                         inst✝ : NonUnitalNonAssocRing R
                         S : Set (TwoSidedIdeal R)
                         x : R
                         ⊢ Iff (∀ (x_1 : RingCon R), Membership.mem (Set.image TwoSidedIdeal.ringCon S) …
                       -/
  show (∀ _, _) ↔ _ by simp [mem_iff]
                       /-
                         🎉 no goals
                       -/


instance : Top (TwoSidedIdeal R) where
  top := { ringCon := ⊤ }


lemma top_ringCon : (⊤ : TwoSidedIdeal R).ringCon = ⊤ := rfl


@[simp]
lemma mem_top {x : R} : x ∈ (⊤: TwoSidedIdeal R) := trivial


instance : Bot (TwoSidedIdeal R) where
  bot := { ringCon := ⊥ }


lemma bot_ringCon : (⊥ : TwoSidedIdeal R).ringCon = ⊥ := rfl


@[simp]
lemma mem_bot {x : R} : x ∈ (⊥ : TwoSidedIdeal R) ↔ x = 0 :=
  Iff.rfl


instance : CompleteLattice (TwoSidedIdeal R) where
  __ := (inferInstance : SemilatticeSup (TwoSidedIdeal R))
  __ := (inferInstance : SemilatticeInf (TwoSidedIdeal R))
  __ := (inferInstance : CompleteSemilatticeSup (TwoSidedIdeal R))
  __ := (inferInstance : CompleteSemilatticeInf (TwoSidedIdeal R))
                 /-
                   R : Type u_1
                   inst✝ : NonUnitalNonAssocRing R
                   x✝ : TwoSidedIdeal R
                   ⊢ LE.le x✝ Top.top
                 -/
  le_top _ := by rw [ringCon_le_iff]; exact le_top
                                      /-
                                        🎉 no goals
                                      -/
                 /-
                   R : Type u_1
                   inst✝ : NonUnitalNonAssocRing R
                   x✝ : TwoSidedIdeal R
                   ⊢ LE.le Bot.bot x✝
                 -/
  bot_le _ := by rw [ringCon_le_iff]; exact bot_le
                                      /-
                                        🎉 no goals
                                      -/


lemma one_mem_iff {R : Type*} [NonAssocRing R] (I : TwoSidedIdeal R) :
    (1 : R) ∈ I ↔ I = ⊤ :=
                                       /-
                                         R : Type u_2
                                         inst✝ : NonAssocRing R
                                         I : TwoSidedIdeal R
                                         h : Membership.mem I 1
                                         x : R
                                         x✝ : Membership.mem Top.top x
                                         ⊢ Membership.mem I x
                                       -/
  ⟨fun h => eq_top_iff.2 fun x _ => by simpa using I.mul_mem_left x _ h, fun h ↦ h.symm ▸ trivial⟩
                                       /-
                                         🎉 no goals
                                       -/


alias ⟨eq_top, one_mem⟩ := one_mem_iff


