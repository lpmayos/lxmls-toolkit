import pdb

from lxmls.sequences.id_feature import IDFeatures


# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    # def add_initial_features(self, sequence, y, features):
    #     return features
    #
    # def add_final_features(self, sequence, y_prev, features):
    #     return features

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        x_name = self.dataset.x_dict.get_label_name(x)
        word = x_name
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if word.istitle():
            # Generate feature name.
            feat_name = "uppercased::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if word.isdigit():
            # Generate feature name.
            feat_name = "number::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if word.find("-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in range(max_prefix):
            if len(word) > i+1:
                prefix = word[:i+1]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        return features

    def add_transition_features(self, sequence, pos, y, y_prev, features, y_next=None):
        """ Adds a feature to the edge feature list.
        Creates a unique id if its the first time the feature is visited
        or returns the existing id otherwise
        """
        # assert pos < len(sequence.x) , pdb.set_trace()

        x = sequence.x[pos]

        # Get label name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get previous label name from ID.
        y_prev_name = self.dataset.y_dict.get_label_name(y_prev)

        # Generate feature name.
        feat_name = "prev_tag:%s::%s" % (y_prev_name, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if y_next is not None:
            y_next_name = self.dataset.y_dict.get_label_name(y_next)
            # Generate feature name.
            feat_name = "prev_and_next_tag:%s::%s::%s" % (y_prev_name, y_name, y_next_name)
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Get word name from ID.
        x_name = self.dataset.x_dict.get_label_name(x)
        word = x_name

        # not working! (does not give any improvement)
        # if word == "'s":
        #     # Generate feature name.
        #     feat_name = "verb_here:%s::%s" % (y_prev_name, y_name)
        #     # Get feature ID from name.
        #     feat_id = self.add_feature(feat_name)
        #     # Append feature.
        #     if feat_id != -1:
        #         features.append(feat_id)

        return features

# with initial and final features (3 epochs)
# CRF - Extended Features Accuracy Train: 0.418 Dev: 0.430 Test: 0.467

# with initial and final features + verb_here (3 epochs)
# CRF - Extended Features Accuracy Train: 0.418 Dev: 0.430 Test: 0.467

# with all + prev_and_next_tag (3 epochs)
# CRF - Extended Features Accuracy Train: 0.945 Dev: 0.890 Test: 0.866

# with all + prev_and_next_tag (20 epochs)
# CRF - Extended Features Accuracy Train: 0.984 Dev: 0.899 Test: 0.894
